from typing import Tuple, List, Union

import numpy as np
from scipy.linalg import sqrtm
import torch as th
import torchvision


def weights_init_normal(m):
    """Initialize weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            th.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find("BatchNorm2d") != -1:
        th.nn.init.normal_(m.weight.data, 1.0, 0.02)
        th.nn.init.constant_(m.bias.data, 0.0)


@th.no_grad()
def vgg16_get_activation_maps(imgs: th.Tensor, layer_idx: int, device: str, normalize_range: Tuple[int, int], global_pooling: bool = True, use_bn: bool = True) -> th.Tensor:
    """Get activation maps of VGG-16 with BN.

    Parameters
    ----------
    imgs : th.Tensor
        images (bs, 3, H, W)
    layer_idx : int
        which layer to return
    device : str
        which device to use
    global_pooling : bool
        apply global max pooling, by default True

    Returns
    -------
    th.Tensor
        activation maps
    """
    vgg16 = torchvision.models.vgg16(pretrained=True).to(
        device) if not use_bn else torchvision.models.vgg16_bn(pretrained=True).to(device)
    vgg16.eval()
    features = vgg16.features

    # normalize imgs for VGG-16:
    min_, max_ = normalize_range
    imgs_norm = (imgs - min_) / (max_ - min_)
    x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(imgs_norm)

    x = x.to(device)
    for i, fet in enumerate(features):
        if i+1 > layer_idx:
            break
        x = fet(x)

    if global_pooling:
        x = th.nn.AdaptiveMaxPool2d(1)(x)

    del vgg16

    return x.detach().cpu().clone()


def fid_score(imgs1: th.Tensor, imgs2: th.Tensor, n_cases: int = 1024, layer_idx: int = 24, device: str = "cpu", normalize_range: Tuple[int, int] = (-1, 1), use_bn: bool = True) -> float:
    """Computes the FID score between ``imgs1`` and ``imgs2`` using VGG-16.

    Parameters
    ----------
    imgs1 : th.Tensor
        Input image 1
    imgs2 : th.Tensor
        Input image 2
    n_cases : int, optional
        number of cases to compute FID score, by default 1024
    layer_idx : int, optional
        vgg16 layer index, by default 24
    device : str, optional
        which device to use, by default "cpu"

    Returns
    -------
    float
        FID score

    Raises
    ------
    RuntimeError
        If the number of ``imgs1`` or ``imgs2`` less than ``n_cases``

    References
    ----------
    Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. 
    2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.
    Neural Information Processing Systems (NIPS)
    """
    imgs1_n_cases = imgs1.shape[0]
    imgs2_n_cases = imgs2.shape[0]

    if imgs1_n_cases < n_cases or imgs2_n_cases < n_cases:
        raise RuntimeError(
            f"At least {n_cases} of images need to be provided but, number of imgs1 is {imgs1_n_cases} and number of imgs2 is {imgs2_n_cases}.")

    # activation of dataset 1
    act1 = np.squeeze(vgg16_get_activation_maps(
        imgs1[:n_cases], layer_idx, device, normalize_range, use_bn=use_bn).numpy())

    # activation of dataset 2
    act2 = np.squeeze(vgg16_get_activation_maps(
        imgs2[:n_cases], layer_idx, device, normalize_range, use_bn=use_bn).numpy())

    # https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        print("[WARNING] Computation of covariance is unstable.")
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
