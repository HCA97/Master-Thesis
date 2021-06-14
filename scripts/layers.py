import torch.nn as nn
import torch as th


class LinearLayer(nn.Module):
    def __init__(self,
                 in_f,
                 out_f,
                 use_bn=True,
                 act="leakyrelu",
                 n_blocks=1,
                 bn_mode="old"):
        super().__init__()

        self.linear_blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.linear_blocks.append(
                nn.Linear(in_f if i == 0 else out_f, out_f, bias=not use_bn))
            if use_bn:
                if bn_mode == "old":
                    self.linear_blocks.append(nn.BatchNorm1d(out_f, eps=0.8))
                elif bn_mode == "momentum":
                    self.linear_blocks.append(
                        nn.BatchNorm1d(out_f, momentum=0.8))
                else:
                    self.linear_blocks.append(nn.BatchNorm1d(out_f))

            if act == "leakyrelu":
                self.linear_blocks.append(nn.LeakyReLU(0.2, inplace=True))
            elif act == "relu":
                self.linear_blocks.append(nn.ReLU(inplace=True))
            elif act == "tanh":
                self.linear_blocks.append(nn.Tanh())
            elif act == "sigmoid":
                self.linear_blocks.append(nn.Sigmoid())
            elif act == "linear":
                pass
            else:
                raise NotImplementedError(f"{act} is not implemented.")

    def forward(self, x):
        for layer in self.linear_blocks:
            x = layer(x)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight

    References
    ----------
    https://github.com/huangzh13/StyleGAN.pytorch/blob/bce838ecfa34d4de69429af4f7e028b63e52c3fe/models/CustomLayers.py#L183
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = th.randn(x.size(0), 1, x.size(
                2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, act="leakyrelu", inject_noise=False, bn_mode="old"):
        super().__init__()
        self.projection = None
        if in_f != out_f:
            self.projection = ConvBlock(
                in_f, out_f, act=act, inject_noise=inject_noise, bn_mode=bn_mode)
        self.conv_block = nn.Sequential(
            ConvBlock(out_f, out_f, act=act, inject_noise=inject_noise, bn_mode=bn_mode), ConvBlock(out_f, out_f, act="linear", bn_mode=bn_mode))
        self.act = nn.LeakyReLU(
            0.2, inplace=True) if act == "leakyrelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        if self.projection is not None:
            x = self.projection(x)
        x = self.act(x + self.conv_block(x))
        return x


class ConvBlock(nn.Module):
    """Simple convolution block.

    [conv -> bn (if use_bn) -> act -> dropout (if dropout)] x n_blocks

    Parameters
    ----------
    in_f : int
        Input filter size
    out_f : int
        Output filter size
    kernel_size : int
        Kernel size, by default 3
    stride : int
        Conv stride, by default 1
    dropout : bool
        Use drop out (it uses dropout 0.25), by default False
    use_bn : bool
        Use batch normalization, by default True
    act : str
        Non-linearity options are `relu, leakyrelu, tanh, sigmoid`, by default leakyrelu
    padding : int
        by default 1
    n_blocks : int
        by default 1

    Notes
    -----
    If ``kernel_size`` is 1 then ``padding`` set to 0

    Raises
    ------
    NotImplementedError
    """

    def __init__(self, in_f, out_f, kernel_size=3, stride=1, dropout=False, use_bn=True, act="leakyrelu", padding=1, n_blocks=1, inject_noise=False, bn_mode="old"):
        """ConvBlocks Constructor"""
        super(ConvBlock, self).__init__()

        if kernel_size == 1:
            padding = 0

        self.conv_block = nn.ModuleList()
        for i in range(n_blocks):
            self.conv_block.append(
                nn.Conv2d(in_f if i == 0 else out_f, out_f, kernel_size, stride=stride, padding=padding, bias=not use_bn))
            if use_bn:
                if bn_mode == "old":
                    self.conv_block.append(nn.BatchNorm2d(out_f, eps=0.8))
                elif bn_mode == "momentum":
                    self.conv_block.append(nn.BatchNorm2d(out_f, momentum=0.8))
                else:
                    self.conv_block.append(nn.BatchNorm2d(out_f))

            if act == "leakyrelu":
                self.conv_block.append(nn.LeakyReLU(0.2, inplace=True))
                # self.conv_block.append(nn.LeakyReLU(inplace=True))
            elif act == "relu":
                self.conv_block.append(nn.ReLU(inplace=True))
            elif act == "tanh":
                self.conv_block.append(nn.Tanh())
            elif act == "sigmoid":
                self.conv_block.append(nn.Sigmoid())
            elif act == "linear":
                pass
            else:
                raise NotImplementedError(f"{act} is not implemented.")

            if dropout:
                self.conv_block.append(nn.Dropout2d(0.25))

            if inject_noise:
                self.conv_block.append(NoiseLayer(out_f))

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x
