import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch
import cv2
import glob
import PIL

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecLocoDataset(Dataset):
    def __init__(self, root_dir, category, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.category = category
        self.image_paths = sorted(glob.glob(root_dir+f"{self.category}/*"))
        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image = self.transform_img(imageo)
        Image = {'image':image,'image1':image1}
        return Image

class MVTecLocoLogicalDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"test/logical_anomalies/*"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)



    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image = self.transform_img(imageo)
        Image = {'image':image,'image1':image1}
        return Image

class MVTecLocoTestGoodDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"test/good/*"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)



    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image = self.transform_img(imageo)
        Image = {'image':image,'image1':image1}
        return Image

class MVTecLocoTestStruDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"test/structural_anomalies/*"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image = self.transform_img(imageo)
        Image = {'image':image,'image1':image1}
        return Image

class MVTecLocoTestValDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"test/validation/*"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image = self.transform_img(imageo)
        Image = {'image':image,'image1':image1}
        return Image