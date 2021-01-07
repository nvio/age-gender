import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils.pretrained import remove_last_layers
import torch.nn.functional as F


class AgeGenderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        resnet = remove_last_layers(resnet, 1)
        self.feature_extractor = nn.Sequential(resnet, nn.Flatten())
        self.fc_age = nn.Sequential(nn.Linear(512, 7), nn.LeakyReLU(0.2))
        self.fc_gender = nn.Sequential(nn.Linear(512, 2), nn.LeakyReLU(0.2))

    @classmethod
    def create(cls):
        return cls()

    def forward(self, image):
        features = self.feature_extractor(image)
        age = F.softmax(self.fc_age(features), dim=1)
        gender = F.softmax(self.fc_gender(features), dim=1)
        return age, gender


if __name__ == "__main__":
    batch = torch.zeros((2, 3, 200, 200)).to("cuda")
    model = AgeGenderResNet().to("cuda")
    model(batch)