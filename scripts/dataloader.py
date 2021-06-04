from typing import Tuple, List, Union
import os

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from pytorch_lightning.core.lightning import LightningModule


class MNISTDataModule(LightningModule):
    """MNIST data loader for GAN. It is used for testing the gan."""

    def __init__(self, img_size: Tuple[int, int] = (32, 32), data_dir: str = "./data/mnist", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = None

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = datasets.MNIST(self.data_dir,
                                      train=True, download=True,
                                      transform=transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                                                    transforms.Normalize([0.5], [0.5])]))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)


class ImageFolder(Dataset):
    """docstring for ArtDataset
    https://github.com/odegeasslbc/FastGAN-pytorch"""

    def __init__(self, root, transform=None):
        super(ImageFolder, self).__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # if self.use_cuda:
        #     img = img.cuda()
        return img, img


class PostdamCarsDataModule(LightningModule):
    """Potsdam cars data loader for GAN."""

    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (32, 64), batch_size: int = 64, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = None
        self.transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        if transform:
            self.transform = transform

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = ImageFolder(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
