import torch.nn as nn


class ConvBlock(nn.Module):
    """Simple convolution block.

    conv -> bn (if use_bn) -> act -> dropout (if dropout)

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
    """

    def __init__(self, in_f, out_f, stride=1, dropout=False, use_bn=True, act="leakyrelu"):
        """ConvBlock Constructor"""
        super(ConvBlock, self).__init__()

        self.conv_block = nn.ModuleList()
        self.conv_block.append(
            nn.Conv2d(in_f, out_f, 3, stride=stride, padding=1, bias=not use_bn))
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
        else:
            raise NotImplementedError(f"{act} is not implemented.")

        if dropout:
            self.conv_block.append(nn.Dropout2d(0.25))

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x