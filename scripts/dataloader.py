from typing import Tuple, List, Union
import os
import random

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core.lightning import LightningModule


class ImageFolder(Dataset):
    """Image folder dataset

    Reference
    ---------
    https://github.com/odegeasslbc/FastGAN-pytorch
    """

    def __init__(self, root1, root2="", transform1=None, transform2=None):
        super(ImageFolder, self).__init__()
        self.frame1 = self._parse_frame(root1)
        self.frame2 = self._parse_frame(root2) if root2 else []
        self.transform1 = transform1
        self.transform2 = transform2

    def _parse_frame(self, root):
        frame = []
        img_names = os.listdir(root)
        random.shuffle(img_names)
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return max(len(self.frame1), len(self.frame2))

    def __getitem__(self, idx):

        # main file
        file1 = self.frame1[idx % len(self.frame1)]
        img1 = Image.open(file1).convert('RGB')

        if self.transform1:
            img1 = self.transform1(img1)

        # second file
        if self.frame2:
            file2 = self.frame2[idx % len(self.frame2)]
            img2 = Image.open(file2).convert('RGB')
            if self.transform2:
                img2 = self.transform2(img2)
            return img1, img2
        return img1, img1


class PostdamCarsDataModule(LightningModule):
    """Potsdam cars data loader for GAN."""

    def __init__(self, data_dir1: str, data_dir2: str = "", img_size: Tuple[int, int] = (32, 64), batch_size: int = 64, transform1=None, transform2=None):
        super().__init__()
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = None
        self.transform1 = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        self.transform2 = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        if transform1:
            self.transform1 = transform1
        if transform2:
            self.transfrom2 = transform2

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = ImageFolder(
            self.data_dir1, root2=self.data_dir2, transform1=self.transform1, transform2=self.transform2)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
