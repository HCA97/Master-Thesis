import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, act="leakyrelu"):
        super().__init__()
        self.projection = None
        if in_f != out_f:
            self.projection = ConvBlock(in_f, out_f, act=act)
        self.conv_block = nn.Sequential(
            ConvBlock(out_f, out_f, act=act), ConvBlock(out_f, out_f, act="linear"))
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

    def __init__(self, in_f, out_f, kernel_size=3, stride=1, dropout=False, use_bn=True, act="leakyrelu", padding=1, n_blocks=1):
        """ConvBlocks Constructor"""
        super(ConvBlock, self).__init__()

        if kernel_size == 1:
            padding = 0

        self.conv_block = nn.ModuleList()
        for i in range(n_blocks):
            self.conv_block.append(
                nn.Conv2d(in_f if i == 0 else out_f, out_f, kernel_size, stride=stride, padding=padding, bias=not use_bn))
            if use_bn:
                # self.conv_block.append(nn.BatchNorm2d(out_f, 0.8))
                # self.conv_block.append(nn.BatchNorm2d(out_f, momentum=0.8))
                self.conv_block.append(nn.BatchNorm2d(out_f))

            if act == "leakyrelu":
                # self.conv_block.append(nn.LeakyReLU(0.2, inplace=True))
                self.conv_block.append(nn.LeakyReLU(inplace=True))
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

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x
