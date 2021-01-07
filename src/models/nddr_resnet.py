import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet101
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
        

class NDDR_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.nddr_layers = nn.ModuleList([NDDRLayer(64), NDDRLayer(64), NDDRLayer(128), NDDRLayer(256), NDDRLayer(512)])
        self.age_net = self.get_resnet_modulelist()
        self.gender_net = self.get_resnet_modulelist()
        self.fc_age = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                    nn.Flatten(),
                                    nn.Linear(512, 7),
                                    nn.LeakyReLU(0.2)
        )
        self.fc_gender = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                       nn.Flatten(),
                                       nn.Linear(512, 2),
                                       nn.LeakyReLU(0.2)
        )

    @classmethod
    def create(cls):
        return cls()

    def get_resnet_modulelist(self):
        model = remove_last_layers(resnet18(pretrained=True), 2)
        model = nn.ModuleList([nn.Sequential(*list(model.children())[:4]), *list(model.children())[4:]])
        return model


    def forward(self, x):
        x_age = x
        x_gender = x
        for nddr_layer, age_layer, gender_layer in zip(self.nddr_layers, self.age_net, self.gender_net):
            x_age = age_layer(x_age)
            x_gender = gender_layer(x_gender)
            x_age, x_gender = nddr_layer(x_age, x_gender)

        age = torch.softmax(self.fc_age(x_age), dim=1)
        gender = torch.softmax(self.fc_gender(x_gender), dim=1)

        return age, gender

if __name__ == "__main__":
    batch = torch.zeros((2, 3, 200, 200)).to("cuda")
    model = NDDR_ResNet18().to("cuda")
    model(batch)
