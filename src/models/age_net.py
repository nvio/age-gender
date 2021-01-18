import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50, mobilenet_v2
from utils.confusion_matrix import confusion_matrix_plot_as_array
from utils.pretrained import remove_last_layers
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_lightning.metrics.functional.classification import accuracy


class AgeNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.feature_extractor, num_features = self.adapt_backbone(backbone)
        self.fc_age = nn.Sequential(nn.Linear(num_features, 7), nn.LeakyReLU(0.2))

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


    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.softmax(self.fc_age(x), dim=1)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.005)

    def training_step(self, batch, batch_idx):
        output = self.compute_output(batch, stage="train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.compute_output(batch, stage="val")
        return output

    def compute_output(self, batch, stage):
        image, age_group, gender = batch
        age_probs = self(image)
        loss = F.nll_loss(torch.log(age_probs), age_group)
        self.log_dict({f"{stage}/loss/age": loss})
        output = {"loss": loss,
                  "age": (age_group, age_probs.argmax(dim=1))
                  }
        return output

    def training_epoch_end(self, outputs):
        if self.is_time_to_log():
            self.log_epoch_results(outputs, stage="train")

    def validation_epoch_end(self, outputs):
        if self.is_time_to_log():
            self.log_epoch_results(outputs, stage="val")

    def is_time_to_log(self, log_freq=10):
        return self.current_epoch % log_freq == 0

    def log_epoch_results(self, outputs, stage):
        age_pred = []
        age_true = []
        for i, output in enumerate(outputs):
            age_true.extend(self.tensor_to_array(output["age"][0]))
            age_pred.extend(self.tensor_to_array(output["age"][1]))

        age_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60-"]
        cm_age = confusion_matrix_plot_as_array(age_true, age_pred, age_labels)

        # Tensorboard logs
        age_accuracy = accuracy(torch.tensor(age_pred).to(self.device), (torch.tensor(age_true).to(self.device))).item()

        tb = self.logger.experiment
        tb.add_scalar(f"{stage}/Age accuracy", age_accuracy, self.current_epoch)
        tb.add_image(f"{stage}/Age", cm_age, self.current_epoch, dataformats="HWC")

    @staticmethod
    def tensor_to_array(tensor):
        return np.array(tensor.cpu(), dtype=np.uint8)

