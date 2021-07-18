import math
from typing import List, Union, Tuple


import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..layers import *


# ------------------------------------------------------- #
#              GAN Losses                                 #
# ------------------------------------------------------- #

class L2Norm(nn.Module):
    def forward(self, x1, x2):
        return th.mean((x1 - x2)**2)

# ------------------------------------------------------- #
#           GAN INVERSION                                 #
# ------------------------------------------------------- #


class EncoderLatent(nn.Module):
    def __init__(self, generator: nn.Module, img_dim=(3, 32, 64), base_channels=32, n_layers=4, n_blocks=1):
        super().__init__()

        self.generator = generator
        for param in self.generator.parameters():
            param.requires_grad = False

        self.conv_blocks = []

        for i in range(n_layers):
            self.conv_blocks.append(ConvBlock(img_dim[0] if i == 0 else base_channels * 2**i,
                                              base_channels * 2**(i+1),
                                              bn_mode="default",
                                              use_bn=True,
                                              n_blocks=n_blocks))
            self.conv_blocks.append(nn.MaxPool2d(2, 2))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        output_width, output_height = img_dim[1], img_dim[2]
        for i in range(n_layers):
            output_width = math.ceil((output_width - 2 + 2*padding) / 2)
            output_height = math.ceil((output_height - 2 + 2*padding) / 2)
        ds_size = output_width * output_height
        self.l1 = nn.Linear(ds_size*base_channels * 2 **
                            (n_layers+1), self.generator.latent_dim)

    def forward(self, img):
        img = self.conv_blocks(img)
        img = img.view(img.shape[0], -1)
        z = self.l1(img)
        img_rec = self.generator(z)
        return z, img_rec


# -------------------------------------------------------- #
#                   Discriminators                         #
# -------------------------------------------------------- #

class MultiDiscriminator(nn.Module):
    """https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/munit/models.py#L197"""

    def __init__(self, img_size, n_res=3, kernel_size=4, base_channels=64, use_instance_norm=True, use_sigmoid=False, use_spectral_norm=True, padding_mode="zeros", use_dropout=False):
        super().__init__()

        input_channels, input_height, input_width = img_size

        self.n_res = n_res

        # IMAGE DOMAIN 1
        self.multi_res1 = nn.ModuleList(
            [PatchDiscriminator(img_size,
                                base_channels=base_channels,
                                kernel_size=kernel_size,
                                padding=int((kernel_size - 1) / 2),
                                use_instance_norm=use_instance_norm,
                                use_sigmoid=use_sigmoid,
                                use_spectral_norm=use_spectral_norm,
                                padding_mode=padding_mode,
                                use_dropout=use_dropout)
             for _ in range(n_res)]
        )

        # IMAGE DOMAIN 2
        self.multi_res2 = nn.ModuleList(
            [PatchDiscriminator(img_size,
                                base_channels=base_channels,
                                kernel_size=kernel_size,
                                padding=int((kernel_size - 1) / 2),
                                use_instance_norm=use_instance_norm,
                                use_sigmoid=use_sigmoid,
                                use_spectral_norm=use_spectral_norm,
                                padding_mode=padding_mode,
                                use_dropout=use_dropout)
             for _ in range(n_res)]
        )

        self.downsample = nn.AvgPool2d(3, padding=1, stride=2)

    def forward(self, x, label):
        ret = []
        for disc1, disc2 in zip(self.multi_res1,  self.multi_res2):
            ret.append(disc1(x) if label == 1 else disc2(x))
            x = self.downsample(x)
        return ret


class BasicPatchDiscriminator(nn.Module):
    def __init__(self,
                 img_size,
                 base_channels=64,
                 n_layers=4,
                 bn_mode="default",
                 use_instance_norm=False,
                 use_sigmoid=True,
                 use_dropout=True,
                 use_spectral_norm=False,
                 padding_mode="reflect",
                 use_bn_first_conv=False):
        super().__init__()

        self.basic = BasicDiscriminator(img_size,
                                        base_channels=base_channels,
                                        n_layers=n_layers,
                                        bn_mode=bn_mode,
                                        use_instance_norm=use_instance_norm,
                                        use_sigmoid=use_sigmoid,
                                        use_dropout=use_dropout,
                                        use_spectral_norm=use_spectral_norm,
                                        padding_mode=padding_mode,
                                        use_bn_first_conv=use_bn_first_conv)

        self.patch = PatchDiscriminator(img_size,
                                        base_channels=base_channels,
                                        n_layers=n_layers,
                                        bn_mode=bn_mode,
                                        use_instance_norm=use_instance_norm,
                                        use_sigmoid=use_sigmoid,
                                        use_dropout=use_dropout,
                                        use_spectral_norm=use_spectral_norm,
                                        padding_mode=padding_mode,
                                        use_bn_first_conv=use_bn_first_conv)

    def forward(self, x):
        x1 = self.basic(x)
        x2 = self.patch(x)
        return [x1, x2]


class PatchDiscriminator(nn.Module):
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

    def __init__(self,
                 img_size,
                 base_channels=16,
                 kernel_size=3,
                 n_layers=4,
                 padding=1,
                 bn_mode="old",
                 use_instance_norm=False,
                 use_sigmoid=True,
                 use_dropout=True,
                 use_spectral_norm=False,
                 padding_mode="zeros",
                 use_bn_first_conv=False):
        super().__init__()

        self.n_layers = n_layers

        input_channels, input_height, input_width = img_size

        layers = [ConvBlock(input_channels,
                            base_channels,
                            stride=2,
                            dropout=use_dropout,
                            padding=padding,
                            use_instance_norm=False,
                            use_spectral_norm=use_spectral_norm,
                            use_bn=use_bn_first_conv,
                            kernel_size=kernel_size,
                            padding_mode=padding_mode,
                            bn_mode=bn_mode)]
        for i in range(1, n_layers):
            layers.append(ConvBlock(base_channels*2**(i-1),
                                    base_channels*2**i,
                                    stride=2,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    dropout=use_dropout,
                                    use_instance_norm=use_instance_norm,
                                    use_bn=not use_instance_norm,
                                    bn_mode=bn_mode,
                                    padding_mode=padding_mode,
                                    use_spectral_norm=use_spectral_norm))

        self.conv_blocks = nn.Sequential(*layers)
        self.l1 = ConvBlock(base_channels*2**(n_layers-1),
                            1,
                            kernel_size=3,
                            padding=1,
                            use_instance_norm=False,
                            use_bn=False,
                            padding_mode=padding_mode,
                            act="sigmoid" if use_sigmoid else "linear")

    def forward(self, x, **kwargs):
        x = self.conv_blocks(x)
        x = self.l1(x)
        return x


class ResNetDiscriminator(nn.Module):
    def __init__(self,
                 img_size=(3, 32, 64),
                 base_channels=64,
                 n_layers=4,
                 use_spectral_norm=False,
                 use_instance_norm=False,
                 use_dropout=True,
                 n_blocks=1,
                 padding_mode="reflect",
                 bn_mode="default",
                 use_sigmoid=True,
                 **kwargs):
        super().__init__()

        self.n_layers = n_layers

        input_channels, input_height, input_width = img_size

        layers = [ConvBlock(input_channels,
                            base_channels,
                            dropout=use_dropout,
                            use_instance_norm=False,
                            use_bn=False,
                            padding_mode=padding_mode,
                            bn_mode=bn_mode)]
        for i in range(n_layers):
            layers.append(ResBlock(base_channels*2**(i-1 if i > 0 else i),
                                   base_channels*2**i,
                                   dropout=use_dropout,
                                   use_instance_norm=use_instance_norm,
                                   use_bn=not use_instance_norm,
                                   bn_mode=bn_mode,
                                   padding_mode=padding_mode,
                                   use_spectral_norm=use_spectral_norm))
            for _ in range(1, n_blocks):
                layers.append(ResBlock(base_channels*2**i,
                                       base_channels*2**i,
                                       dropout=use_dropout,
                                       use_instance_norm=use_instance_norm,
                                       use_bn=not use_instance_norm,
                                       bn_mode=bn_mode,
                                       padding_mode=padding_mode,
                                       use_spectral_norm=use_spectral_norm))
            layers.append(nn.AvgPool2d(2, 2))

        self.conv_blocks = nn.Sequential(*layers)

        output_width, output_height = input_width, input_height
        for i in range(n_layers):
            output_width = math.ceil(output_width / 2)
            output_height = math.ceil(output_height / 2)
        ds_size = output_width * output_height
        self.l1 = nn.Sequential(
            nn.Linear(base_channels*2**(n_layers-1) * ds_size, 1),
            nn.Sigmoid() if use_sigmoid else nn.Identity())

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        return x


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

    def __init__(self,
                 img_size,
                 base_channels=16,
                 n_layers=4,
                 heat_map=False,
                 heat_map_layer=4,
                 padding=1,
                 use_instance_norm=False,
                 use_dropout=True,
                 use_sigmoid=True,
                 bn_mode="old",
                 use_spectral_norm=False,
                 padding_mode="zeros",
                 use_bn_first_conv=False):
        super().__init__()

        self.heat_map = heat_map
        self.heat_map_layer = min(heat_map_layer, n_layers)
        self.n_layers = n_layers

        input_channels, input_height, input_width = img_size

        layers = [ConvBlock(input_channels,
                            base_channels,
                            stride=2,
                            dropout=use_dropout,
                            use_instance_norm=False,
                            use_bn=use_bn_first_conv,
                            padding_mode=padding_mode,
                            bn_mode=bn_mode)]
        for i in range(1, n_layers):
            layers.append(ConvBlock(base_channels*2**(i-1),
                                    base_channels*2**i,
                                    stride=2,
                                    padding=padding,
                                    dropout=use_dropout,
                                    use_instance_norm=use_instance_norm,
                                    use_bn=not use_instance_norm,
                                    bn_mode=bn_mode,
                                    padding_mode=padding_mode,
                                    use_spectral_norm=use_spectral_norm))

        self.conv_blocks = nn.Sequential(*layers)
        if self.heat_map:
            self.patch = ConvBlock(base_channels*2**(self.heat_map_layer-1),
                                   1,
                                   padding_mode=padding_mode,
                                   padding=padding,
                                   use_bn=False,
                                   act="sigmoid" if use_sigmoid else "linear")

        output_width, output_height = input_width, input_height
        for i in range(n_layers):
            output_width = math.ceil((output_width - 2 + 2*padding) / 2)
            output_height = math.ceil((output_height - 2 + 2*padding) / 2)
        ds_size = output_width * output_height
        self.l1 = nn.Sequential(
            nn.Linear(base_channels*2**(n_layers-1) * ds_size, 1),
            nn.Sigmoid() if use_sigmoid else nn.Identity())

    def forward(self, x, **kwargs):
        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if self.heat_map and self.heat_map_layer == i+1:
                p = self.patch(x)
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        if self.heat_map:
            return [x, p]
        return x


# -------------------------------------------------------- #
#                       Generators                         #
# -------------------------------------------------------- #


class MUNITGEN(nn.Module):
    def __init__(self, img_dim, base_channels=64, use_spectral_norm=True, style_dim=8, act="relu", n_layers=3, padding_mode="zeros"):
        super().__init__()

        self.encoder1 = MUNITEncoder(
            img_dim, base_channels=base_channels, use_spectral_norm=use_spectral_norm, act=act, style_dim=style_dim, padding_mode=padding_mode, n_layers=n_layers)
        self.decoder1 = MUNITDecoder(
            img_dim, base_channels=base_channels, use_spectral_norm=use_spectral_norm, act=act, n_layers=n_layers, padding_mode=padding_mode)

        self.encoder2 = MUNITEncoder(
            img_dim, base_channels=base_channels, use_spectral_norm=use_spectral_norm, act=act, style_dim=style_dim, n_layers=n_layers, padding_mode=padding_mode)
        self.decoder2 = MUNITDecoder(
            img_dim, base_channels=base_channels, use_spectral_norm=use_spectral_norm, act=act, n_layers=n_layers, padding_mode=padding_mode)

        self.style_dim = style_dim

    def forward(self, x1, x2, inference=False):

        c1, s1 = self.encoder1(x1)
        s1_ = th.normal(0, 1, size=(
            x1.size(0), self.style_dim), device=x1.device)

        c2, s2 = self.encoder2(x2)
        s2_ = th.normal(0, 1, size=(
            x2.size(0), self.style_dim), device=x2.device)

        x11 = self.decoder1(c1, s1)
        x22 = self.decoder1(c2, s2)

        x12 = self.decoder2(c1, s1_)
        x21 = self.decoder1(c2, s2_)

        c12, s12 = self.encoder2(x12)
        c21, s21 = self.encoder1(x21)

        x121 = self.decoder1(c12, s1)
        x212 = self.decoder2(c21, s2)

        if inference:
            return [x12, x21, x121, x212, x11, x22]

        # compute all the regularization
        loss_rec = th.mean(th.abs(x11 - x1)) + th.mean(th.abs(x22 - x2))
        loss_s = th.mean(th.abs(s12 - s1_)) + th.mean(th.abs(s21 - s2_))
        loss_c = th.mean(th.abs(c12 - c1.detach())) + \
            th.mean(th.abs(c21 - c2.detach()))
        loss_cyc = th.mean(th.abs(x121 - x1)) + th.mean(th.abs(x212 - x2))
        return [x12, x21], [loss_rec, loss_s, loss_c, loss_cyc]


class MUNITDecoder(nn.Module):
    def __init__(self, img_size, base_channels=64, use_spectral_norm=True, style_dim=8, act="relu", n_layers=3, padding_mode="zeros"):
        super().__init__()

        input_channels, input_height, input_width = img_size

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(4*base_channels, 2*base_channels,
                      kernel_size=5,
                      padding=2,
                      act=act,
                      use_bn=False,
                      use_instance_norm=False,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock(2*base_channels, base_channels,
                      kernel_size=5,
                      padding=2,
                      act=act,
                      use_bn=False,
                      use_instance_norm=False,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            ConvBlock(base_channels, input_channels,
                      kernel_size=7,
                      padding=3,
                      act="tanh",
                      use_bn=False,
                      use_instance_norm=False,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),

        )

        self.mlps1 = nn.ModuleList([LinearLayer(
            style_dim, 2*4*base_channels, use_bn=False, act=act, n_blocks=3) for i in range(n_layers)])
        self.mlps2 = nn.ModuleList([LinearLayer(
            style_dim, 2*4*base_channels, use_bn=False, act=act, n_blocks=3) for i in range(n_layers)])

        self.resblocks1 = nn.ModuleList(
            [nn.Conv2d(4*base_channels, 4*base_channels, 3,
                       padding=1, bias=False, padding_mode=padding_mode) for i in range(n_layers)]
        )
        self.resblocks2 = nn.ModuleList(
            [nn.Conv2d(4*base_channels, 4*base_channels, 3,
                       padding=1, bias=False, padding_mode=padding_mode) for i in range(n_layers)]
        )

        # normalization and non linearity
        self.ad = AdaIN()
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, c, s):

        for mlp1, mlp2, conv1, conv2 in zip(self.mlps1, self.mlps2, self.resblocks1, self.resblocks2):
            x = conv1(c)
            y = mlp1(s)
            x = self.ad(x, y)
            x = self.act(x)

            x = conv2(x)
            y = mlp2(s)
            x = self.ad(x, y)
            c = self.act(x + c)

        # print(c.shape)

        x = self.upsample(c)
        # print(x.shape)
        return x


class MUNITEncoder(nn.Module):
    def __init__(self, img_size, base_channels=64, use_spectral_norm=True, style_dim=8, act="relu", n_layers=3, padding_mode="zeros"):
        super().__init__()

        input_channels, input_height, input_width = img_size

        self.content_enc = [
            ConvBlock(input_channels, base_channels,
                      kernel_size=7,
                      use_instance_norm=True,
                      use_bn=False,
                      padding=3,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            ConvBlock(base_channels, 2*base_channels,
                      kernel_size=4,
                      stride=2,
                      use_instance_norm=True,
                      use_bn=False,
                      padding=1,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            ConvBlock(2*base_channels, 4*base_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      use_instance_norm=True,
                      use_bn=False,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm)
        ]

        self.content_enc += [
            ResBlock(4*base_channels, 4*base_channels,
                     act=act,
                     use_instance_norm=True,
                     padding_mode=padding_mode,
                     use_spectral_norm=use_spectral_norm) for _ in range(n_layers)
        ]

        self.content_enc = nn.Sequential(*self.content_enc)

        self.style_enc = [
            ConvBlock(input_channels, base_channels,
                      kernel_size=7,
                      use_instance_norm=True,
                      use_bn=False,
                      padding=3,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            ConvBlock(base_channels, 2*base_channels,
                      kernel_size=4,
                      stride=2,
                      use_instance_norm=True,
                      use_bn=False,
                      padding=1,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm),
            ConvBlock(2*base_channels, 4*base_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      use_instance_norm=True,
                      use_bn=False,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm)
        ]
        n_times = 1 if min(input_height, input_width) < 64 else 2
        self.style_enc += [
            ConvBlock(4*base_channels, 4*base_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      use_instance_norm=True,
                      use_bn=False,
                      act=act,
                      padding_mode=padding_mode,
                      use_spectral_norm=use_spectral_norm) for _ in range(n_times)
        ]
        self.style_enc.append(nn.AdaptiveAvgPool2d(1))

        self.style_enc = nn.Sequential(*self.style_enc)

        self.style_enc_l1 = nn.Linear(4*base_channels, style_dim)

    def forward(self, x):
        c = self.content_enc(x)
        s = self.style_enc(x)
        s = s.view(x.shape[0], -1)
        s = self.style_enc_l1(s)
        return c, s


class RefinerNet(nn.Module):
    def __init__(self,
                 n_channels=3,
                 init_channels=128,
                 n_layers=4,
                 act="relu",
                 bn_mode="old",
                 reconstruct=False,
                 use_spectral_norm=False,
                 inject_noise=False,
                 padding_mode="zeros",
                 **kwargs):
        super().__init__()

        self.reconstruct = reconstruct

        layers = [ResBlock(n_channels, init_channels,
                           act=act, bn_mode=bn_mode, use_spectral_norm=use_spectral_norm)]
        for _ in range(n_layers-1):
            if inject_noise:
                layers.append(NoiseLayer(init_channels))

            layers.append(
                ResBlock(init_channels, init_channels, act=act, bn_mode=bn_mode, padding_mode=padding_mode, use_spectral_norm=use_spectral_norm))

        self.conv_blocks = nn.Sequential(*layers)
        self.final = ConvBlock(
            init_channels, 3, use_bn=False, act="tanh", bn_mode=bn_mode, padding_mode=padding_mode)

    def forward(self, x_in):

        x = self.conv_blocks(x_in)
        x = self.final(x)

        if self.reconstruct:
            return x
        return x_in + x


class UnetGenerator(nn.Module):

    def __init__(self,
                 n_channels=3,
                 init_channels=64,
                 n_layers=4,
                 n_blocks=2,
                 act="relu",
                 bn_mode="old",
                 reconstruct=False,
                 use_spectral_norm=False,
                 inject_noise=False,
                 use_bn_first_conv=True,
                 padding_mode="zeros",
                 **kwargs):
        super().__init__()

        if n_layers < 2:
            raise AttributeError("Number of layers must be larger than 1.")

        self.reconstruct = reconstruct

        # down part
        self.down = nn.ModuleList()
        self.down.append(
            ConvBlock(n_channels, init_channels,
                      n_blocks=n_blocks, bn_mode=bn_mode, act=act, use_bn=use_bn_first_conv,
                      use_spectral_norm=use_spectral_norm, padding_mode=padding_mode))
        for i in range(1, n_layers-1):
            self.down.append(
                ConvBlock(2**(i-1) * init_channels, 2**i * init_channels,
                          n_blocks=n_blocks, act=act, bn_mode=bn_mode,
                          use_spectral_norm=use_spectral_norm, padding_mode=padding_mode))

        # bottleneck
        self.bottle_neck = ConvBlock(
            2**(n_layers-2) * init_channels, 2**(n_layers-1) * init_channels,
            n_blocks=n_blocks, act=act, bn_mode=bn_mode,
            use_spectral_norm=use_spectral_norm, inject_noise=inject_noise, padding_mode=padding_mode)

        # up part
        self.up = nn.ModuleList()
        for i in range(n_layers-1, 0, -1):
            block = nn.Sequential(
                ConvBlock((2**(i - 1) + 2**i)*init_channels,
                          2**(i-1) * init_channels, kernel_size=1, act=act, padding_mode=padding_mode,
                          bn_mode=bn_mode, use_spectral_norm=use_spectral_norm),
                ConvBlock(2**(i-1) * init_channels, 2**(i-1)
                          * init_channels, n_blocks=n_blocks, inject_noise=inject_noise, padding_mode=padding_mode,
                          act=act, bn_mode=bn_mode, use_spectral_norm=use_spectral_norm)
            )
            self.up.append(block)

        self.final = ConvBlock(
            init_channels, n_channels, use_bn=False, act="tanh", bn_mode=bn_mode, padding_mode=padding_mode)

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
        if self.reconstruct:
            return x
        return x_in + x


class ResNetGenerator(nn.Module):
    def __init__(self,
                 img_size=(3, 32, 64),
                 latent_dim=100,
                 init_channels=512,
                 n_layers=4,
                 act="leakyrelu",
                 n_blocks=1,
                 bn_mode="default",
                 inject_noise=False,
                 use_spectral_norm=False,
                 padding_mode="reflect",
                 **kwargs):
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

        for n in range(n_blocks):
            layers.append(ResBlock(self.init_channels,
                                   self.init_channels,
                                   padding_mode=padding_mode,
                                   act=act,
                                   bn_mode=bn_mode,
                                   inject_noise=inject_noise,
                                   use_spectral_norm=use_spectral_norm))

        # middle layer
        for i in range(1, n_layers):
            layers.append(nn.Upsample(scale_factor=2))

            layers.append(ResBlock(self.init_channels // 2 ** (i-1),
                                   self.init_channels // 2 ** i,
                                   padding_mode=padding_mode,
                                   act=act,
                                   inject_noise=inject_noise,
                                   bn_mode=bn_mode,
                                   use_spectral_norm=use_spectral_norm))

            for n in range(1, n_blocks):
                layers.append(ResBlock(self.init_channels // 2 ** i,
                                       self.init_channels // 2 ** i,
                                       inject_noise=inject_noise,
                                       act=act,
                                       bn_mode=bn_mode,
                                       padding_mode=padding_mode,
                                       use_spectral_norm=use_spectral_norm))

        # last layer
        layers.append(ConvBlock(self.init_channels // 2**(n_layers - 1),
                                input_channels,
                                use_bn=False,
                                use_spectral_norm=False,
                                padding_mode=padding_mode,
                                act="tanh",
                                bn_mode=bn_mode))

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x


class BasicGenerator(nn.Module):
    """Basic Generator. 

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

    def __init__(self,
                 img_size,
                 latent_dim=100,
                 init_channels=128,
                 n_layers=2,
                 act="leakyrelu",
                 inject_noise=False,
                 bn_mode="old",
                 use_spectral_norm=False,
                 last_layer_kernel_size=3,
                 padding_mode="zeros",
                 kernel_size=3,
                 **kwargs):

        super().__init__()

        input_channels, input_height, input_width = img_size

        scale_factor = 2**n_layers
        if input_height % scale_factor != 0 or input_width % scale_factor != 0:
            raise AttributeError(
                f"Input size ({input_width}, {input_height}) must be divisible by scale factor {scale_factor}.")

        if last_layer_kernel_size % 2 == 0 or kernel_size % 2 == 0:
            raise AttributeError(
                f"Kernel size for last layer ({last_layer_kernel_size}) and other layers ({kernel_size}) must be odd number!")

        self.init_height = input_height // scale_factor
        self.init_width = input_width // scale_factor
        self.latent_dim = latent_dim
        self.init_channels = init_channels

        self.l1 = nn.Linear(self.latent_dim, self.init_channels *
                            self.init_width * self.init_height)

        # first layer
        layers = [nn.BatchNorm2d(self.init_channels),
                  nn.Upsample(scale_factor=2)]
        layers.append(ConvBlock(self.init_channels,
                                self.init_channels,
                                act=act,
                                bn_mode=bn_mode,
                                inject_noise=inject_noise,
                                padding_mode=padding_mode,
                                padding=int((kernel_size - 1) / 2),
                                kernel_size=kernel_size,
                                use_spectral_norm=use_spectral_norm))

        # middle layer
        for i in range(1, n_layers):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(ConvBlock(self.init_channels // 2 ** (i-1),
                                    self.init_channels // 2 ** i,
                                    act=act, bn_mode=bn_mode,
                                    inject_noise=inject_noise,
                                    padding_mode=padding_mode,
                                    kernel_size=kernel_size,
                                    padding=int((kernel_size - 1) / 2),
                                    use_spectral_norm=use_spectral_norm))

        # last layer
        layers.append(ConvBlock(self.init_channels // 2**(n_layers-1),
                                input_channels,
                                kernel_size=last_layer_kernel_size,
                                padding=int((last_layer_kernel_size - 1) / 2),
                                use_bn=False,
                                use_spectral_norm=False,
                                act="tanh",
                                padding_mode=padding_mode,
                                bn_mode=bn_mode))

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x
