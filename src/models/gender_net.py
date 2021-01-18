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


class GenderNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.feature_extractor, num_features = self.adapt_backbone(backbone)
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

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.softmax(self.fc_gender(x), dim=1)
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
        image, _, gender = batch
        gender_probs = self(image)
        loss = F.nll_loss(torch.log(gender_probs), gender)
        self.log_dict({f"{stage}/loss/gender": loss})
        output = {"loss": loss,
                  "gender": (gender, gender_probs.argmax(dim=1))
                  }
        return output

    def log_epoch_results(self, outputs, stage):
        gender_pred = []
        gender_true = []
        for i, output in enumerate(outputs):
            gender_true.extend(self.tensor_to_array(output["gender"][0]))
            gender_pred.extend(self.tensor_to_array(output["gender"][1]))

        cm_gender = confusion_matrix_plot_as_array(gender_true, gender_pred, ["Male", "Female"])

        # Tensorboard logs
        gender_accuracy = accuracy(torch.tensor(gender_pred).to(self.device), (torch.tensor(gender_true).to(self.device))).item()

        tb = self.logger.experiment
        tb.add_scalar(f"{stage}/Gender accuracy", gender_accuracy, self.current_epoch)
        tb.add_image(f"{stage}/Gender", cm_gender, self.current_epoch, dataformats="HWC")

    def training_epoch_end(self, outputs):
        if self.is_time_to_log():
            self.log_epoch_results(outputs, stage="train")

    def validation_epoch_end(self, outputs):
        if self.is_time_to_log():
            self.log_epoch_results(outputs, stage="val")

    def is_time_to_log(self, log_freq=10):
        return self.current_epoch % log_freq == 0


    @staticmethod
    def tensor_to_array(tensor):
        return np.array(tensor.cpu(), dtype=np.uint8)
