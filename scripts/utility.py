import torch


def weights_init_normal(m):
    """Initialize weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
