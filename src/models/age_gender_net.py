import sys

sys.path.append(r"D:\projects\ulta\src")

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, mobilenet_v2
from utils.pretrained import remove_last_layers
import torch.nn.functional as F
import pytorch_lightning as pl

class AgeGenderNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        backbone, num_features = self.adapt_backbone(backbone)
        self.feature_extractor = backbone
        self.fc_age = nn.Sequential(nn.Linear(num_features, 7), nn.LeakyReLU(0.2))
        self.fc_gender = nn.Sequential(nn.Linear(num_features, 2), nn.LeakyReLU(0.2))

    @classmethod
    def create(cls, cfg):
        pretrained = cfg.pretrained
        if cfg.backbone == "resnet18":
            backbone = resnet18(pretrained)
        elif cfg.backbone == "resnet50":
            backbone = resnet50(pretrained)
        elif cfg.backbone == "mobilenet_v2":
            backbone = mobilenet_v2(pretrained)
        else:
            raise ValueError(f"{cfg.backbone} is not a valid backbone")
        return cls(backbone)

    @staticmethod
    def adapt_backbone(backbone):
        num_features = list(backbone.modules())[-1].in_features
        backbone = remove_last_layers(backbone, 1)
        last_layer = list(backbone.modules())[-1]
        if not isinstance(last_layer, nn.AdaptiveAvgPool2d):
            backbone = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        else:
            backbone = nn.Sequential(backbone, nn.Flatten())
        return backbone, num_features


    def forward(self, image):
        features = self.feature_extractor(image)
        age = F.softmax(self.fc_age(features), dim=1)
        gender = F.softmax(self.fc_gender(features), dim=1)
        return age, gender


if __name__ == "__main__":
    batch = torch.zeros((2, 3, 200, 200)).to("cuda")
    backbone = resnet50(pretrained=True)
    model = AgeGenderNet(backbone).to("cuda")
    model(batch)