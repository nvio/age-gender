import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import DataModule, UTKFaceDataset
import torch.nn as nn
import torch

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
        
        return loss




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
