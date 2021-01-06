import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional.classification import accuracy

from utils.confusion_matrix import confusion_matrix_plot_as_array


from dataloader import DataModule, UTKFaceDataset
from models.cnn_agegender import AgeGenderResNet

class Net(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_age = nn.NLLLoss()
        self.loss_gender = nn.NLLLoss()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        output = self.compute_output(batch, stage="train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.compute_output(batch, stage="val")
        return output

    def compute_output(self, batch, stage):
        image, age_group, gender = batch
        age_group_probs, gender_probs = self.net(image)
        age_loss = self.loss_age(torch.log(age_group_probs), age_group)
        gender_loss = self.loss_gender(torch.log(gender_probs), gender)
        loss = age_loss + gender_loss
        self.log_dict({f"{stage}/loss/age": age_loss, f"{stage}/loss/gender": gender_loss, f"{stage}/loss/total": loss})
        output = {"loss": loss,
                  "age": (age_group, age_group_probs.argmax(dim=1)),
                  "gender": (gender, gender_probs.argmax(dim=1))}
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
        gender_pred = []
        gender_true = []
        for i, output in enumerate(outputs):
            age_true.extend(self.tensor_to_array(output["age"][0]))
            age_pred.extend(self.tensor_to_array(output["age"][1]))
            gender_true.extend(self.tensor_to_array(output["gender"][0]))
            gender_pred.extend(self.tensor_to_array(output["gender"][1]))

        age_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60-"]
        cm_age = confusion_matrix_plot_as_array(age_true, age_pred, age_labels)
        cm_gender = confusion_matrix_plot_as_array(gender_true, gender_pred, ["Male", "Female"])

        # Tensorboard logs
        age_accuracy = accuracy(torch.tensor(age_pred).to(self.device), (torch.tensor(age_true).to(self.device))).item()
        gender_accuracy = accuracy(torch.tensor(gender_pred).to(self.device), (torch.tensor(gender_true).to(self.device))).item()

        tb = self.logger.experiment
        tb.add_scalar(f"{stage}/Age accuracy", age_accuracy, self.current_epoch)
        tb.add_scalar(f"{stage}/Gender accuracy", gender_accuracy, self.current_epoch)
        tb.add_image(f"{stage}/Age", cm_age, self.current_epoch, dataformats="HWC")
        tb.add_image(f"{stage}/Gender", cm_gender, self.current_epoch, dataformats="HWC")


    @staticmethod
    def tensor_to_array(tensor):
        return np.array(tensor.cpu(), dtype=np.uint8)



if __name__ == "__main__":


    checkpoint = ModelCheckpoint(monitor='val/loss/total', save_top_k=1)
    logger = TensorBoardLogger(save_dir=r"..\training", name="AgeGenderResNet")
    trainer = pl.Trainer(gpus=1,
                         callbacks=[checkpoint],
                         logger=logger)


    datamodule = DataModule("..\\data\\UTKFace", batch_size=4)
    net = AgeGenderResNet()
    model = Net(net)
    trainer.fit(model, datamodule)
