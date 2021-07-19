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

    def __init__(self, root, root2="", transform=None, transform2=None, shuffle=True):
        super(ImageFolder, self).__init__()
        self.transform1 = transform
        self.transform2 = transform2
        self.shuffle = shuffle
        self.frame1 = self._parse_frame(root)
        self.frame2 = self._parse_frame(root2) if root2 else []

    def _parse_frame(self, root):
        frame = []
        img_names = os.listdir(root)
        if self.shuffle:
            random.shuffle(img_names)
        for img_name in img_names:
            image_path = os.path.join(root, img_name)
            if os.path.splitext(image_path)[-1] in ['.jpg', '.png', '.jpeg']:
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

    def __init__(self, data_dir: str, data_dir2: str = "", img_size: Tuple[int, int] = (32, 64), batch_size: int = 64, transform=None, transform2=None, shuffle=True):
        super().__init__()
        self.data_dir1 = data_dir
        self.data_dir2 = data_dir2
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.dataset = None
        self.transform1 = transforms.Compose([transforms.Resize(self.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])])
        self.transform2 = transforms.Compose([transforms.Resize(self.img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])])
        if transform:
            self.transform1 = transform
        if transform2:
            self.transform2 = transform2

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = ImageFolder(
            self.data_dir1, root2=self.data_dir2, transform=self.transform1, transform2=self.transform2, shuffle=self.shuffle)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
