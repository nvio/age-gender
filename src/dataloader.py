import os
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.dataset import Dataset, random_split
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomAffine


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


    def imshow(self, index, apply_transforms=True):
        image_path = self.image_paths[index]
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        if apply_transforms:
            image = np.array(self.transforms(image).permute(1,2,0))
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

    def plot_statistic(self):
        n_images  = len(self)
        ages = np.zeros(n_images, dtype=np.uint8)
        age_groups_count = np.zeros(7, dtype=np.int32)
        genders_count = np.zeros(2, dtype=np.uint8)
        for i, image_path in enumerate(self.image_paths):
            age, gender = self.get_age_and_gender(image_path)
            ages[i] = age
            age_group = self.get_age_group(age)
            age_groups_count[age_group] += 1
            genders_count[gender] += 1
        
        plt.figure()
        plt.title("Gender distribution")
        plt.bar(np.arange(2), height=genders_count)
        plt.xticks(np.arange(2), ["Male", "Female"])
        plt.ylabel("N")

        plt.figure()
        plt.title("Age distribution")
        plt.hist(ages)

        plt.figure()
        plt.title("Age groups distribution")
        plt.bar(np.arange(len(age_groups_count)), height=age_groups_count)
        plt.xticks(np.arange(len(age_groups_count)), ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60-"])
        plt.ylabel("N")
        plt.show()



class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        transforms=Compose([ToTensor(),
                            RandomHorizontalFlip(),
                            RandomAffine(degrees=5,
                                         translate=(0.05, 0.05),
                                         scale=(1., 1.5))])

        self.dataset = UTKFaceDataset(data_dir, transforms)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def create(cls, config):
        data_dir = config["path"]
        batch_size = config["batch"]
        num_workers = config["num_workers"]
        return cls(data_dir, batch_size, num_workers)

    
    def setup(self, stage=None):
        train_len = int(0.7*len(self.dataset))
        val_len = int(0.2*len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        self.train, self.val, self.test = random_split(self.dataset, 
                                                       (train_len, val_len, test_len),
                                                       generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":
    # Examples
    dataset = UTKFaceDataset("..\\data\\UTKFace",
                            transforms=Compose([ToTensor(), 
                                                RandomHorizontalFlip(), 
                                                RandomAffine(degrees=5,
                                                            translate=(0.05, 0.05),
                                                            scale=(1., 1.5))]))

    dataset.imshow(0)
    dataset.plot_statistic()