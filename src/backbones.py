"""Frozen pretrained video backbones.

Main model : VideoMAE v2-Base   (Hugging Face "OpenGVLab/VideoMAEv2-Base", remote code)
Baseline   : R(2+1)D-18         (torchvision, Kinetics-400 weights)

Input  : clips from the frozen dataset pipeline, uint8 tensor (B, 16, 3, 224, 224)
         = (batch, time, colour, height, width), already resized (short side 256) and centre-cropped (224).
Output : float32 features, (B, 768) for VideoMAE v2 and (B, 512) for R(2+1)D-18.

This file only contains the model-specific steps: colour normalisation, tensor layout and the forward pass.
The backbones are frozen: we never train them, we only use them as fixed feature extractors
(this is the point of the project, and it lets us compute the features once and cache them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = ["videomae_v2", "r2plus1d_18"]
VIDEOMAE_CHECKPOINT = "OpenGVLab/VideoMAEv2-Base"
R2PLUS1D_WEIGHTS = "KINETICS400_V1"
FEATURE_DIM = {"videomae_v2": 768, "r2plus1d_18": 512}

# R(2+1)D-18 was trained on 112x112 inputs. The existing experiment down-scales the shared 224x224 crop
# to 112x112 with trilinear interpolation. This is kept unchanged so the cached features and results stay valid.
R2PLUS1D_INPUT_SIZE = 112

# Each model must see colours normalised the way it was pretrained: ImageNet statistics for VideoMAE v2,
# Kinetics-400 statistics for R(2+1)D-18. Shape (1, 3, 1, 1, 1) broadcasts over (B, 3, T, H, W).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def get_device():
    """GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_backbone(name):
    """Settings that define a backbone's features. Stored in every cache file and checked when it is reused."""
    if name == "videomae_v2":
        return {"backbone": name, "weights": VIDEOMAE_CHECKPOINT, "normalization": "imagenet",
                "model_input": "3x16x224x224", "precision": "fp16_autocast"}
    if name == "r2plus1d_18":
        return {"backbone": name, "weights": "torchvision:" + R2PLUS1D_WEIGHTS, "normalization": "kinetics400",
                "model_input": "3x16x112x112_trilinear_from_224", "precision": "fp16_autocast"}
    raise ValueError(f"Unknown backbone '{name}', expected one of {BACKBONES}")


def load_backbone(name, device):
    """Load the pretrained backbone, put it on `device`, switch to eval mode and freeze every weight."""
    if name == "videomae_v2":
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(VIDEOMAE_CHECKPOINT, trust_remote_code=True)
        model = AutoModel.from_pretrained(VIDEOMAE_CHECKPOINT, config=config, trust_remote_code=True)
    elif name == "r2plus1d_18":
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        model.fc = nn.Identity()          # remove the Kinetics-400 classifier, keep the 512-d feature
    else:
        raise ValueError(f"Unknown backbone '{name}', expected one of {BACKBONES}")

    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False   # frozen: the backbone is a fixed feature extractor
    return model


@torch.no_grad()
def encode_clips(name, model, clips_uint8):
    """Extract one feature vector per clip.

    clips_uint8 : uint8 tensor (B, 16, 3, 224, 224) on the model's device
    returns     : float32 tensor (B, 768) for videomae_v2, (B, 512) for r2plus1d_18
    """
    device = clips_uint8.device
    use_fp16 = device.type == "cuda"
    num_frames = clips_uint8.shape[1]

    x = clips_uint8.float() / 255.0                 # (B, T, 3, H, W), values in [0, 1]
    x = x.permute(0, 2, 1, 3, 4).contiguous()       # video-model layout: (B, 3, T, H, W) = (B, 3, 16, 224, 224)

    if name == "videomae_v2":
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            # extract_features = mean of all patch tokens followed by the final LayerNorm (fc_norm)
            features = model.extract_features(x)                                   # (B, 768)
        return features.float()

    if name == "r2plus1d_18":
        with torch.autocast(device_type=device.type, enabled=use_fp16):
            x = F.interpolate(x, size=(num_frames, R2PLUS1D_INPUT_SIZE, R2PLUS1D_INPUT_SIZE),
                              mode="trilinear", align_corners=False)               # (B, 3, 16, 112, 112)
            x = (x - KINETICS_MEAN.to(device)) / KINETICS_STD.to(device)
            features = model(x)                                                     # (B, 512)
        return features.float()

    raise ValueError(f"Unknown backbone '{name}', expected one of {BACKBONES}")
