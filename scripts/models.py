import torch.nn as nn

from .layers import *
from .utility import *


class BasicDiscriminator(nn.Module):
    """Basic Discriminator. It is similar to the
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

    Parameters
    ----------
    img_size : list or tuple
        Input image size (input_channels, input_height, input_width)

    References
    ----------
    Alec Radford and Luke Met and Soumith Chintala. 2016.
    UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.
    """

    def __init__(self, img_size):
        super().__init__()

        input_channels, input_height, input_width = img_size
        self.model = nn.Sequential(
            ConvBlock(input_channels, 16, stride=2,
                      dropout=True, use_bn=False),
            ConvBlock(16, 32, stride=2, dropout=True),
            ConvBlock(32, 64, stride=2, dropout=True),
            ConvBlock(64, 128, stride=2, dropout=True)
        )

        self.ds_size = (input_width // 2**4) * (input_height // 2**4)
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * self.ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        x = self.model(img)
        x = x.view(x.shape[0], 128 * self.ds_size)
        x = self.adv_layer(x)
        return x


class BasicGenerator(nn.Module):
    """Basic Generator. It is similar to the 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

    Parameters
    ----------
    img_size : list or tuple
        Input image size (input_channels, input_height, input_width)
    latent_dim : int
        Latent dimension, by default 100

    References
    ----------
    Alec Radford and Luke Met and Soumith Chintala. 2016. 
    UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.
    """

    def __init__(self, img_size, latent_dim=100, init_channels=128):
        super().__init__()

        input_channels, input_height, input_width = img_size
        self.init_height = input_height // 4
        self.init_width = input_width // 4
        self.latent_dim = latent_dim
        self.init_channels = init_channels

        self.l1 = nn.Linear(self.latent_dim, self.init_channels *
                            self.init_width * self.init_height)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_channels),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            ConvBlock(64, input_channels, use_bn=False, act="tanh")
        )

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x
