import os
import cv2 as cv
from glob import glob
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np



class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.image_paths = glob(data_dir+"/*")
        self.transforms = transforms

        print("UTKFace dataset")
        print(f"  Root: {self.data_dir}")
        print(f"  Image size: 200x200")
        print(f"  Number of images: {len(self)}")


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        age, gender = self.get_age_and_gender(image_path)
        age_group = self.get_age_group(age)
        if self.transforms is not None:
            image = self.transforms(image)

        return (image, age_group, gender)


    def __len__(self):
        return len(self.image_paths)


    def imshow(self, index):
        image_path = self.image_paths[index]
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        age, gender = self.get_age_and_gender(image_path)
        image = cv.putText(image, 
                            text = f"Gender: {'Female' if gender==1 else 'Male'}, Age: {age}",
                            org=(5, image.shape[1]-5), 
                            color=(255, 255, 255), 
                            fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.4)
        plt.figure()
        plt.imshow(image)
        plt.show()


    @staticmethod
    def get_age_and_gender(image_path):
        image_name = os.path.basename(image_path)
        age, gender = image_name.split("_")[:2]
        age = int(age)
        gender = int(gender)
        return age, gender

    @staticmethod
    def get_age_group(age):
        group = None
        if age <= 10:
            group = 0
        elif age <= 20:
            group = 1
        elif age <= 30:
            group = 2
        elif age <= 40:
            group = 3
        elif age <= 50:
            group = 4
        elif age <= 60:
            group = 5 
        else:
            group = 6

        assert group != None

        return group



class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        transforms = Compose([ToTensor()])
        self.dataset = UTKFaceDataset(data_dir, transforms)
        self.batch_size = batch_size
        

    def setup(self, stage=None):
        self.train, self.val, self.test = random_split(self.dataset, 
                                                       (np.array([0.7, 0.2, 0.1])*len(self.dataset)).astype(np.uint8),
                                                       generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)

if __name__ == "__main__":
    # Examples
    dataset = UTKFaceDataset("..\\data\\UTKFace")
