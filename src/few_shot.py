"""Few-shot classification on cached features: episodes, ProtoHead, training and evaluation.

Input  : features (N, D) of one split + the class name and source group of every video
         (from feature_extractor.load_split_features).
Output : accuracies (mean % and 95 % confidence interval) and one result dictionary per experiment.
Next   : run_experiments.py calls train_and_evaluate() for every setting and writes the CSV files.

Two ways to build an episode
  group_safe=True  : the query videos of a class never come from a UCF101 source group (gXX) that is used
                     in its support set. This is the protocol we report.
  group_safe=False : "standard_random" protocol (TEAM's): support and query are drawn at random, so a query may
                     share its source group with a support video (group overlap allowed). Kept only for comparison.
In both protocols the same video is never used as support and query in one episode.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FewShotDataset:
    """Features of one split, kept on the GPU, plus what is needed to build episodes."""

    def __init__(self, features, class_names, group_ids, device):
        is_finite = torch.isfinite(features).all(dim=1).tolist()
        self.features = features.to(device)                          # (N, D)
        self.group = []                                              # source video of each row: (class, group)
        for class_name, group_id in zip(class_names, group_ids):
            self.group.append((class_name, group_id))
        self.videos_of_class = {}                                    # class -> row indices (valid videos only)
        for index, class_name in enumerate(class_names):
            if is_finite[index]:
                self.videos_of_class.setdefault(class_name, []).append(index)

    def keep_fraction(self, fraction, min_keep, seed):
        """New dataset keeping `fraction` of the videos of each class (at least `min_keep`)."""
        rng = random.Random(seed)
        smaller = FewShotDataset.__new__(FewShotDataset)
        smaller.features = self.features
        smaller.group = self.group
        smaller.videos_of_class = {}
        for class_name, indices in self.videos_of_class.items():
            num_keep = min(len(indices), max(min_keep, int(round(fraction * len(indices)))))
            smaller.videos_of_class[class_name] = rng.sample(indices, num_keep)
        return smaller

    def pick_group_safe(self, class_name, shot, num_queries, rng, tries=20):
        """Group-safe support/query for one class: draw the support, then draw the queries only among
        videos whose source group is not used by any support video. Returns None if impossible."""
        indices = self.videos_of_class[class_name]
        for _ in range(tries):
            support = rng.sample(indices, shot)
            support_groups = set()
            for index in support:
                support_groups.add(self.group[index])
            candidates = []
            for index in indices:
                if self.group[index] not in support_groups:
                    candidates.append(index)
            if len(candidates) >= num_queries:
                return support, rng.sample(candidates, num_queries)
        return None

    def sample_episode(self, way, shot, num_queries, rng, group_safe):
        """One N-way K-shot episode.

        returns support (way*shot, D), query (way*num_queries, D), query_labels (way*num_queries,)
        Support rows are ordered class by class, which ProtoHead relies on.
        """
        eligible_classes = []
        for class_name, indices in self.videos_of_class.items():
            if len(indices) >= shot + num_queries:
                eligible_classes.append(class_name)
        if len(eligible_classes) < way:
            raise ValueError(f"Only {len(eligible_classes)} classes have enough videos; need {way}.")

        picks = []
        if not group_safe:
            for class_name in rng.sample(eligible_classes, way):
                chosen = rng.sample(self.videos_of_class[class_name], shot + num_queries)   # distinct videos
                picks.append((chosen[:shot], chosen[shot:]))
        else:
            # visit the classes in random order until `way` of them can be split group-safely
            for class_name in rng.sample(eligible_classes, len(eligible_classes)):
                result = self.pick_group_safe(class_name, shot, num_queries, rng)
                if result is not None:
                    picks.append(result)
                if len(picks) == way:
                    break
            if len(picks) < way:
                raise ValueError(f"Only {len(picks)} classes admit a group-disjoint {shot}-shot split; need {way}.")

        support_indices = []
        query_indices = []
        for support, query in picks:
            support_indices.extend(support)
            query_indices.extend(query)
        query_labels = torch.arange(way).repeat_interleave(num_queries)   # [0, .., 0, 1, .., 1, ...]
        return self.features[support_indices], self.features[query_indices], query_labels


class ProtoHead(nn.Module):
    """Prototype classifier with a small trainable projection.

    feature -> Linear(D, D) -> L2 normalisation -> class prototype (mean of the support of each class)
    -> cosine similarity between query and prototypes -> multiplied by a learnable scale = logits.
    The linear layer starts as the identity, so before training ProtoHead is exactly the plain prototype
    classifier on the frozen features ("notrain" columns).
    """

    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))
            self.proj.bias.zero_()
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, support, query, way, shot):
        support = F.normalize(self.proj(support), dim=-1)          # (way*shot, D)
        query = F.normalize(self.proj(query), dim=-1)              # (way*q, D)
        prototypes = support.view(way, shot, -1).mean(dim=1)       # (way, D): average the shots of each class
        prototypes = F.normalize(prototypes, dim=-1)
        return self.scale * query @ prototypes.t()                  # (way*q, way) scaled cosine similarities


@torch.no_grad()
def evaluate_head(head, data, num_tasks, way, shot, num_queries, seed, group_safe):
    """Mean accuracy (%) and 95 % confidence interval over `num_tasks` episodes (TEAM's formula 1.96*std/sqrt(n)*100)."""
    head.eval()
    rng = random.Random(seed)
    accuracies = []
    for _ in range(num_tasks):
        support, query, query_labels = data.sample_episode(way, shot, num_queries, rng, group_safe)
        predictions = head(support, query, way, shot).argmax(-1).cpu()
        accuracies.append((predictions == query_labels).float().mean().item())
    accuracies = np.array(accuracies)
    return accuracies.mean() * 100.0, 196.0 * accuracies.std() / np.sqrt(len(accuracies))


def learning_rate_at(iteration, cfg):
    """TEAM's step schedule: LR x 1, 0.5, 0.1, 0.01 from epochs 0, 3, 5, 7 (1 epoch = iters_per_epoch iterations)."""
    epoch = iteration / cfg["iters_per_epoch"]
    boundaries = cfg["lr_steps"] + [cfg["max_epoch"]]
    for index, boundary in enumerate(boundaries):
        if epoch < boundary:          # first boundary above the current epoch; the factor before it applies
            break
    return cfg["lr"] * cfg["lr_factors"][index - 1]


def train_and_evaluate(train_data, val_data, test_data, way, shot, train_frac, cfg, device):
    """One complete experiment.

    1. test the untrained ProtoHead (= plain prototypes) on the test classes
    2. train ProtoHead with episodes from the training classes (SGD, TEAM schedule)
    3. every `val_every` iterations evaluate on the validation classes and keep the best head
    4. test the best head
    Returns one result row (dictionary).
    """
    set_seed(cfg["seed"])
    train_data = train_data.keep_fraction(train_frac, shot + cfg["queries_train"], cfg["seed"])

    num_test_classes = 0
    for indices in test_data.videos_of_class.values():
        if len(indices) >= shot + cfg["queries_test"]:
            num_test_classes += 1
    eval_way = min(way, num_test_classes)

    head = ProtoHead(train_data.features.shape[1]).to(device)

    def test(group_safe):
        return evaluate_head(head, test_data, cfg["test_tasks"], eval_way, shot, cfg["queries_test"], cfg["seed"],
                             group_safe)

    notrain_standard_random = test(group_safe=False)
    notrain_group_safe = test(group_safe=True)

    optimizer = torch.optim.SGD(list(head.parameters()), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4, nesterov=True)
    rng = random.Random(cfg["seed"])
    best_val = -1
    best_iter = 0
    best_state = copy.deepcopy(head.state_dict())
    for iteration in range(cfg["train_iters"]):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(iteration, cfg)
        head.train()
        support, query, query_labels = train_data.sample_episode(way, shot, cfg["queries_train"], rng,
                                                                 cfg["group_safe"])
        loss = F.cross_entropy(head(support, query, way, shot), query_labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % cfg["val_every"] == 0:
            val_accuracy, _ = evaluate_head(head, val_data, cfg["val_tasks"], min(way, 10), shot, cfg["queries_test"],
                                            cfg["seed"], cfg["group_safe"])
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_iter = iteration + 1
                best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)

    trained_standard_random = test(group_safe=False)
    trained_group_safe = test(group_safe=True)

    videos_per_class = []
    for indices in train_data.videos_of_class.values():
        videos_per_class.append(len(indices))

    return {
        "way": eval_way, "shot": shot, "train_frac": train_frac, "chance": 100.0 / eval_way,
        "notrain_standard_random": notrain_standard_random[0],
        "notrain_group_safe": notrain_group_safe[0],
        "trained_standard_random": trained_standard_random[0],
        "trained_standard_random_ci95": trained_standard_random[1],
        "trained_group_safe": trained_group_safe[0],
        "trained_group_safe_ci95": trained_group_safe[1],
        "val_best": best_val, "best_iter": best_iter, "group_safe_train": cfg["group_safe"],
        "n_train_per_class": float(np.mean(videos_per_class)),
    }


def audit_group_safety(test_data, seed, shot=5, num_checks=2000):
    """Check the samplers: group-safe splits must never share a source group (asserted), and measure how often
    the standard_random protocol puts the query in a support group. Returns that percentage."""
    rng = random.Random(seed)
    classes = list(test_data.videos_of_class)
    for _ in range(num_checks):
        class_name = rng.choice(classes)
        result = test_data.pick_group_safe(class_name, shot, 1, rng)
        if result is not None:
            support, query = result
            support_groups = set()
            for index in support:
                support_groups.add(test_data.group[index])
            for index in query:
                assert test_data.group[index] not in support_groups, "group overlap in group-safe episode"

    num_overlap = 0
    for _ in range(num_checks):
        class_name = rng.choice(classes)
        chosen = rng.sample(test_data.videos_of_class[class_name], shot + 1)
        support_groups = set()
        for index in chosen[:shot]:
            support_groups.add(test_data.group[index])
        if test_data.group[chosen[shot]] in support_groups:
            num_overlap += 1
    return 100.0 * num_overlap / num_checks
