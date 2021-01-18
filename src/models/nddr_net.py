import sys
sys.path.append(r"D:\projects\ulta\src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, mobilenet_v2
from utils.pretrained import remove_last_layers

class NDDRLayer(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        in_channels = 2*out_channels
        self.conv1x1_age = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU()
        )
        self.conv1x1_gender = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU()
        )

    def forward(self, x_age, x_gender):
        x = torch.cat((x_age, x_gender), dim=1)
        x_age = self.conv1x1_age(x)
        x_gender = self.conv1x1_gender(x)
        return x_age, x_gender
        

class NDDRNet(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super().__init__()
        self.age_net, self.nddr_layers, num_features = self.adapt_backbone(backbone, pretrained)
        self.gender_net, _, _ = self.adapt_backbone(backbone, pretrained)
        self.fc_age = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(num_features, 7),
                                    nn.LeakyReLU(0.2)
        )
        self.fc_gender = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                       nn.Flatten(),
                                       nn.Linear(num_features, 2),
                                       nn.LeakyReLU(0.2)
        )

    @classmethod
    def create(cls, cfg):
        return cls(cfg.backbone, cfg.pretrained)

    def adapt_backbone(self, backbone, pretrained):
        if backbone == "resnet18":
            model = self.resnet18_as_modulelist(pretrained)
            nddr_layers = self.resnet18_nddr_layers()
            num_features = 512
        elif backbone == "resnet50":
            model= self.resnet50_as_modulelist(pretrained)
            nddr_layers = self.resnet50_nddr_layers()
            num_features = 2048
        elif backbone == "mobilenet_v2":
            model = self.mobilenet_v2_as_modulelist(pretrained)
            nddr_layers = self.mobilenet_v2_nddr_layers()
            num_features = 1280
        else:
            raise ValueError(f"{backbone} is not a valid backbone")

        return model, nddr_layers, num_features

    @staticmethod
    def resnet18_as_modulelist(pretrained):
        model = remove_last_layers(resnet18(pretrained), 2)
        model = nn.ModuleList([nn.Sequential(*list(model.children())[:4]), *list(model.children())[4:]])
        return model

    @staticmethod
    def resnet50_as_modulelist(pretrained):
        model = remove_last_layers(resnet50(pretrained), 2)
        model = nn.ModuleList([nn.Sequential(*list(model.children())[:4]), *list(model.children())[4:]])
        return model

    @staticmethod
    def mobilenet_v2_as_modulelist(pretrained):
        model = nn.ModuleList(remove_last_layers(mobilenet_v2(pretrained), 1).children())[0]
        return model

    @staticmethod
    def resnet18_nddr_layers():
        return nn.ModuleList([NDDRLayer(64), NDDRLayer(64), NDDRLayer(128), NDDRLayer(256), NDDRLayer(512)])

    @staticmethod
    def resnet50_nddr_layers():
        return nn.ModuleList([NDDRLayer(64), NDDRLayer(256), NDDRLayer(512), NDDRLayer(1024), NDDRLayer(2048)])

    @staticmethod
    def mobilenet_v2_nddr_layers():
        return nn.ModuleList([NDDRLayer(32), NDDRLayer(16), NDDRLayer(24), NDDRLayer(24), NDDRLayer(32),
                              NDDRLayer(32), NDDRLayer(32), NDDRLayer(64), NDDRLayer(64), NDDRLayer(64), 
                              NDDRLayer(64), NDDRLayer(96), NDDRLayer(96), NDDRLayer(96), NDDRLayer(160),
                              NDDRLayer(160), NDDRLayer(160), NDDRLayer(320), NDDRLayer(1280)])

    def forward(self, x):
        x_age = x
        x_gender = x
        for nddr_layer, age_layer, gender_layer in zip(self.nddr_layers, self.age_net, self.gender_net):
            x_age = age_layer(x_age)
            x_gender = gender_layer(x_gender)
            x_age, x_gender = nddr_layer(x_age, x_gender)

        age = F.softmax(self.fc_age(x_age), dim=1)
        gender = F.softmax(self.fc_gender(x_gender), dim=1)

        return age, gender

if __name__ == "__main__":
    batch = torch.zeros((2, 3, 200, 200)).to("cuda")
    model = NDDRNet("mobilenet_v2").to("cuda")
    model(batch)
