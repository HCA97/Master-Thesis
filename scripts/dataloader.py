from typing import Tuple, List, Union
import os

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

    def __init__(self, root1, root2="", transform=None):
        super(ImageFolder, self).__init__()
        self.frame1 = self._parse_frame(root1)
        self.frame2 = self._parse_frame(root2) if root2 else []
        self.transform = transform

    def _parse_frame(self, root):
        frame = []
        img_names = os.listdir(root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        if self.frame2:
            return min(len(self.frame2), len(self.frame1))
        return len(self.frame1)

    def __getitem__(self, idx):

        # main file
        file1 = self.frame1[idx]
        img1 = Image.open(file1).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)

        # second file
        if self.frame2:
            file2 = self.frame2[idx]
            img2 = Image.open(file2).convert('RGB')
            if self.transform:
                img2 = self.transform(img2)
            return img1, img2
        return img1, img1


class PostdamCarsDataModule(LightningModule):
    """Potsdam cars data loader for GAN."""

    def __init__(self, data_dir1: str, data_dir2: str = "", img_size: Tuple[int, int] = (32, 64), batch_size: int = 64, transform=None):
        super().__init__()
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = None
        self.transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        if transform:
            self.transform = transform

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = ImageFolder(
            self.data_dir1, root2=self.data_dir2, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
