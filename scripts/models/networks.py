import math
import torch as th
import torch.nn as nn

from ..layers import *

# -------------------------------------------------------- #
#                   Discriminators                         #
# -------------------------------------------------------- #


class BasicDiscriminator(nn.Module):
    """Simple CNN no fancy layers. It is similar to the
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

    Parameters
    ----------
    img_size : list or tuple
        Input image size (input_channels, input_height, input_width)
    base_channels : int
        number channels in first layer
    n_layers : int
        number of layers


    References
    ----------
    Alec Radford and Luke Met and Soumith Chintala. 2016.
    UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.
    """

    def __init__(self, img_size, base_channels=16, n_layers=4, heat_map=False, padding=1):
        super().__init__()

        self.heat_map = heat_map

        input_channels, input_height, input_width = img_size

        layers = [ConvBlock(input_channels, base_channels, stride=2,
                            dropout=True, use_bn=False)]
        for i in range(1, n_layers):
            layers.append(ConvBlock(base_channels*2**(i-1), base_channels*2**i, stride=2, padding=padding,
                          dropout=True))

        self.conv_blocks = nn.Sequential(*layers)
        if self.heat_map:
            self.l1 = nn.Sequential(
                nn.Conv2d(base_channels*2**(n_layers-1), 1, 1), nn.Sigmoid())
        else:
            output_width, output_height = input_width, input_height
            for i in range(n_layers):
                output_width = math.ceil((output_width - 2 + 2*padding) / 2)
                output_height = math.ceil((output_height - 2 + 2*padding) / 2)
            ds_size = output_width * output_height
            self.l1 = nn.Sequential(
                nn.Linear(base_channels*2**(n_layers-1) * ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        x = self.conv_blocks(img)
        if not self.heat_map:
            x = x.view(x.shape[0], -1)
        x = self.l1(x)
        return x

# -------------------------------------------------------- #
#                       Generators                         #
# -------------------------------------------------------- #


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

    def __init__(self, img_size, latent_dim=100, init_channels=128, n_layers=2):
        super().__init__()

        input_channels, input_height, input_width = img_size

        scale_factor = 2**n_layers
        if input_height % scale_factor != 0 or input_width % scale_factor != 0:
            raise AttributeError(
                f"Input size ({input_width}, {input_height}) must be divisible by scale factor {scale_factor}.")

        self.init_height = input_height // scale_factor
        self.init_width = input_width // scale_factor
        self.latent_dim = latent_dim
        self.init_channels = init_channels

        self.l1 = nn.Linear(self.latent_dim, self.init_channels *
                            self.init_width * self.init_height)

        # first layer
        layers = [nn.BatchNorm2d(self.init_channels),
                  nn.Upsample(scale_factor=2),
                  ConvBlock(self.init_channels, self.init_channels)]

        # middle layer
        for i in range(1, n_layers):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(ConvBlock(self.init_channels // 2 **
                          (i-1), self.init_channels // 2**i))

        # last layer
        layers.append(ConvBlock(self.init_channels // 2**(n_layers - 1),
                      input_channels, use_bn=False, act="tanh"))

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x
