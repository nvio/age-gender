import torch
import torch.nn as nn
from torchvision.models import resnet18

def remove_last_layer(model):
    return nn.Sequential(*(list(model.children())[:-1]))

class AgeGenderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        resnet = remove_last_layer(resnet)
        self.feature_extractor = nn.Sequential(resnet, nn.Flatten())
        self.fc_age = nn.Linear(512, 7)
        self.fc_gender = nn.Linear(512, 2)

    def forward(self, image):
        features = self.feature_extractor(image)
        age = torch.softmax(self.fc_age(features), dim=1)
        gender = torch.softmax(self.fc_gender(features), dim=1)
        return age, gender

