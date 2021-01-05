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
        image, age_group, gender = batch

        age_group_probs, gender_probs = self.net(image)
        
        age_loss = self.loss_age(torch.log(age_group_probs), age_group)
        gender_loss = self.loss_gender(torch.log(gender_probs), gender)
        loss = age_loss + gender_loss
        self.log_dict({"Loss/age": age_loss, "Loss/gender": gender_loss, "Loss/total": loss})
        
        output = {"loss": loss,
                  "age": (age_group, age_group_probs.argmax(dim=1)),
                  "gender": (gender, gender_probs.argmax(dim=1))}
        return output


    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
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
            tb = self.logger.experiment
            age_accuracy = accuracy(torch.tensor(age_pred).to(self.device), (torch.tensor(age_true).to(self.device))).item()
            gender_accuracy = accuracy(torch.tensor(gender_pred).to(self.device), (torch.tensor(gender_true).to(self.device))).item()
            tb.add_scalar("Age accuracy", age_accuracy, self.current_epoch)
            tb.add_scalar("Gender accuracy", gender_accuracy, self.current_epoch)
            
            tb.add_image("Age", cm_age, self.current_epoch, dataformats="HWC")
            tb.add_image("Gender", cm_gender, self.current_epoch, dataformats="HWC")


    @staticmethod
    def tensor_to_array(tensor):
        return np.array(tensor.cpu(), dtype=np.uint8)



if __name__ == "__main__":


    checkpoint = ModelCheckpoint(monitor='Loss/total', save_top_k=1)
    logger = TensorBoardLogger(save_dir=r"..\training", name="AgeGenderResNet")
    trainer = pl.Trainer(gpus=1,
                         callbacks=[checkpoint],
                         logger=logger)


    datamodule = DataModule("..\\data\\UTKFace")
    net = AgeGenderResNet()
    model = Net(net)
    trainer.fit(model, datamodule)
