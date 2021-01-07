import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet101
from utils.confusion_matrix import confusion_matrix_plot_as_array
from utils.pretrained import remove_last_layers
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import DataModule

class AgeResNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        num_features = list(backbone.modules())[-1].in_features
        backbone = remove_last_layers(backbone, 1)
        self.feature_extractor = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.fc_age = nn.Sequential(nn.Linear(num_features, 7), nn.LeakyReLU(0.2))

    def get_backbone(backbone, pretrained=True):
        if backbone == "resnet18":
            return resnet18(pretrained)
        elif backbone == "resnet101":
            return resnet101(pretrained)
        else:
            raise ValueError

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
        self.log_dict({f"{stage}/loss": loss})
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


class GenderResNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        num_features = list(backbone.modules())[-1].in_features
        backbone = remove_last_layers(backbone, 1)
        self.feature_extractor = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.fc_gender = nn.Sequential(nn.Linear(num_features, 2), nn.LeakyReLU(0.2))

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
        self.log_dict({f"{stage}/loss": loss})
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


if __name__ == "__main__":
    checkpoint = ModelCheckpoint(monitor='val/loss/gender', save_top_k=1, save_last=True)
    logger = TensorBoardLogger(save_dir=r"..\training", name="GenderResNet")
    trainer = pl.Trainer(gpus=1,
                         max_epochs=100,
                         callbacks=[checkpoint],
                         logger=logger)


    datamodule = DataModule("..\\data\\UTKFace", batch_size=4, num_workers=4)
    backbone = resnet18(pretrained=True)
    model = GenderResNet(backbone)
    trainer.fit(model, datamodule)


        
