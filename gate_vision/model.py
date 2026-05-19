import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class GOAT(nn.Module):

    def __init__(self, num_corners: int = 4):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Adapt first conv from 3-channel RGB to 1-channel grayscale.
        # Initialise by averaging the pretrained RGB weights so we start warm.
        orig = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        self.backbone = backbone.features  # -> [B, 576, 8, 8] for 256×256 input

        self.head = nn.Sequential(
            # Layer 1 - upscale 8->16
            nn.ConvTranspose2d(576, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 2 - upscale 16->32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 3 - upscale 32->64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 4 - upscale 64->128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Channel projection -> 4 corner channels
            nn.Conv2d(32, num_corners, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
