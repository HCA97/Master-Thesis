import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
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
                nn.Conv2d(base_channels*2**(n_layers-1), 1, kernel_size=3, padding=1), nn.Sigmoid())
            # I cannot modify it since pytroch used different names for layers
            # self.l1 = ConvBlock(base_channels*2**(n_layers-1),
            #                     1, padding=padding, use_bn=False, act="sigmoid")
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


class UnetGenerator(nn.Module):

    def __init__(self, n_channels=3, init_channels=64, n_layers=4, n_blocks=2, act="relu", **kwargs):
        super().__init__()

        if n_layers < 2:
            raise AttributeError("Number of layers must be larger than 1.")

        # down part
        self.down = nn.ModuleList()
        self.down.append(
            ConvBlock(n_channels, init_channels, n_blocks=n_blocks))
        for i in range(1, n_layers-1):
            self.down.append(
                ConvBlock(2**(i-1) * init_channels, 2**i * init_channels, n_blocks=n_blocks, act=act))

        # bottleneck
        self.bottle_neck = ConvBlock(
            2**(n_layers-2) * init_channels, 2**(n_layers-1) * init_channels, n_blocks=n_blocks, act=act)

        # up part
        self.up = nn.ModuleList()
        for i in range(n_layers-1, 0, -1):
            block = nn.Sequential(
                ConvBlock((2**(i - 1) + 2**i)*init_channels,
                          2**(i-1) * init_channels, kernel_size=1, padding=0, act=act),
                ConvBlock(2**(i-1) * init_channels, 2**(i-1)
                          * init_channels, n_blocks=n_blocks, act=act)
            )
            self.up.append(block)

        self.final = ConvBlock(init_channels, 3, use_bn=False, act="tanh")

    def forward(self, x_in):

        x = self.down[0](x_in)
        hidden_layers = [x]
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        for layer in self.down[1:]:
            x = layer(x)
            hidden_layers.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottle_neck(x)

        for i, layer in enumerate(self.up):
            x = F.interpolate(x, scale_factor=2)
            x = th.cat([x, hidden_layers[-i-1]], axis=1)
            x = layer(x)

        x = self.final(x)
        return x_in + x


class ResNetGenerator(nn.Module):
    def __init__(self, img_size, latent_dim=100, init_channels=128, n_layers=4, learn_upsample=False, act="relu", n_blocks=1, **kwargs):
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
                  nn.Upsample(scale_factor=2)]
        if learn_upsample:
            layers.append(ConvBlock(self.init_channels,
                          self.init_channels, kernel_size=1, padding=0, act=act))
        for n in range(n_blocks):
            layers.append(ResBlock(self.init_channels,
                          self.init_channels, act=act))

        # middle layer
        for i in range(1, n_layers):
            layers.append(nn.Upsample(scale_factor=2))
            if learn_upsample:
                layers.append(ConvBlock(self.init_channels // 2 ** (i-1),
                                        self.init_channels // 2 ** (i-1), kernel_size=1, padding=0, act=act))
            layers.append(ResBlock(self.init_channels // 2 ** (i-1),
                                   self.init_channels // 2 ** i, act=act))
            for n in range(1, n_blocks):
                layers.append(ResBlock(self.init_channels // 2 ** i,
                                       self.init_channels // 2 ** i, act=act))

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

    def __init__(self, img_size, latent_dim=100, init_channels=128, n_layers=2, act="leakyrelu", learn_upsample=False, inject_noise_conv=False, **kwargs):
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
                  nn.Upsample(scale_factor=2)]
        if learn_upsample:
            layers.append(ConvBlock(self.init_channels,
                          self.init_channels, kernel_size=1, act=act))
        layers.append(ConvBlock(self.init_channels,
                      self.init_channels, act=act))

        # middle layer
        for i in range(1, n_layers):
            layers.append(nn.Upsample(scale_factor=2))
            if learn_upsample:
                layers.append(ConvBlock(self.init_channels // 2 ** (i-1),
                                        self.init_channels // 2 ** (i-1),
                                        kernel_size=1, padding=0, act=act))
            layers.append(ConvBlock(self.init_channels // 2 ** (i-1),
                                    self.init_channels // 2 ** i,
                                    act=act))

        # last layer
        layers.append(ConvBlock(self.init_channels // 2**(n_layers-1),
                      input_channels, use_bn=False, act="tanh"))

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x
