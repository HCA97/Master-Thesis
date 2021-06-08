import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, act="leakyrelu"):
        super().__init__()

        self.conv_block = ConvBlock(in_f, out_f, act=act)
        if in_f != out_f:
            self.conv_block = nn.Sequential(
                ConvBlock(in_f, out_f, act=act), ConvBlock(out_f, out_f, act=act))
        self.skip_connection = ConvBlock(out_f, out_f, act="linear")
        self.act = nn.LeakyReLU(
            0.2, inplace=True) if act == "leakyrelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.act(x + self.skip_connection(x))
        return x


class ConvBlock(nn.Module):
    """Simple convolution block.

    conv -> bn (if use_bn) -> act -> dropout (if dropout) x n_blocks

    Parameters
    ----------
    in_f : int
        Input filter size
    out_f : int
        Output filter size
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

    Raises
    ------
    NotImplementedError
    """

    def __init__(self, in_f, out_f, stride=1, dropout=False, use_bn=True, act="leakyrelu", padding=1, n_blocks=1):
        """ConvBlocks Constructor"""
        super(ConvBlock, self).__init__()

        self.conv_block = nn.ModuleList()
        for i in range(n_blocks):
            self.conv_block.append(
                nn.Conv2d(in_f if i == 0 else out_f, out_f, 3, stride=stride, padding=padding, bias=not use_bn))
            if use_bn:
                self.conv_block.append(nn.BatchNorm2d(out_f, 0.8))

            if act == "leakyrelu":
                self.conv_block.append(nn.LeakyReLU(0.2, inplace=True))
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


# class ConvBlocks(nn.Module):
#     """Multiple simple convolution blocks.


#     Parameters
#     ----------
#     in_f : int
#         Input filter size
#     out_f : int
#         Output filter size
#     stride : int
#         Conv stride, by default 1
#     dropout : bool
#         Use drop out (it uses dropout 0.25), by default False
#     use_bn : bool
#         Use batch normalization, by default True
#     act : str
#         Non-linearity options are `relu, leakyrelu, tanh, sigmoid`, by default leakyrelu
#     """

#     def __init__(self, in_f, out_f, stride=1, dropout=False, use_bn=True, act="leakyrelu", padding=1, n_blocks=1):
#         """ConvBlocks Constructor"""
#         super(ConvBlock, self).__init__()

#         self.conv_block = nn.ModuleList()
#         for i in range(n_blocks):
#             self.conv_block.append(
#                 nn.Conv2d(in_f if i == 0 else out_f, out_f, 3, stride=stride, padding=padding, bias=not use_bn))
#             if use_bn:
#                 self.conv_block.append(nn.BatchNorm2d(out_f, 0.8))

#             if act == "leakyrelu":
#                 self.conv_block.append(nn.LeakyReLU(0.2, inplace=True))
#             elif act == "relu":
#                 self.conv_block.append(nn.ReLU(inplace=True))
#             elif act == "tanh":
#                 self.conv_block.append(nn.Tanh())
#             elif act == "sigmoid":
#                 self.conv_block.append(nn.Sigmoid())
#             elif act == "linear":
#                 pass
#             else:
#                 raise NotImplementedError(f"{act} is not implemented.")

#             if dropout:
#                 self.conv_block.append(nn.Dropout2d(0.25))

#     def forward(self, x):
#         for layer in self.conv_block:
#             x = layer(x)
#         return x
