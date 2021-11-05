from typing import Tuple, List, Union
import math
import os
import re

import numpy as np
import tqdm
from scipy.linalg import sqrtm
import torch as th
import torchvision
import lpips

import PIL
import cv2


class AugmentPipe(th.nn.Module):
    def __init__(self, alpha=0.1, cutout=8, p=0):
        super().__init__()
        self.alpha = alpha
        self.cutout = cutout
        self.p = p

    def forward(self, tensor):

        p = self.p
        cutout = self.cutout

        batch_size = len(tensor)

        if p > 0:
            # noise
            noise = th.normal(0, 1, tensor.shape, device=tensor.device)
            noise = 2*(noise - th.min(noise)) / \
                (th.max(noise) - th.min(noise)) - 1
            idx_noise = np.random.choice(
                batch_size, int(batch_size*p), replace=False)
            alpha = th.zeros([batch_size, 1, 1, 1], device=tensor.device)
            alpha[idx_noise] = self.alpha
            tensor = (1 - alpha)*tensor + alpha*noise

            # cutout
            w, h = tensor.shape[-2:]

            if cutout < w and cutout < h:
                c_w = np.random.choice(w - cutout, batch_size)
                c_h = np.random.choice(h - cutout, batch_size)

                idx_cutout = np.random.choice(
                    batch_size, int(batch_size*p), replace=False)

                mask = th.ones_like(tensor, device=tensor.device)
                for idx in idx_cutout:
                    cw, ch = c_w[idx], c_h[idx]
                    mask[idx, :, cw:cw+cutout, ch:ch+cutout] = 0
                tensor = tensor * mask
        return tensor


def get_epoch_number(checkpoint_path):

    exp = r"epoch=([0-9]+)"

    epochs = []
    for epoch in os.listdir(checkpoint_path):
        m = re.search(exp, epoch)
        if m:
            epochs.append(int(m.groups()[0]))
    epochs.sort()
    return epochs


#
# Transformations
#


class AddNoise(th.nn.Module):
    def __init__(self, alpha=0.1, normalize_range=(-1, 1)):
        super().__init__()
        self.alpha = alpha
        self.min, self.max = normalize_range

    def forward(self, img):
        if type(img) != th.Tensor:
            raise RuntimeError("It must be torch tensor not pil image")

        noise = th.normal(0, 1, size=img.shape, device=img.device)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = ((self.max - self.min) * noise) - ((self.max - self.min) / 2)
        img = (1 - self.alpha) * img + self.alpha * noise
        return img


class DynamicPad(th.nn.Module):
    def __init__(self, min_img_dim=(120, 60), padding_mode="edge", padding_value=255):
        super().__init__()
        self.min_img_dim = min_img_dim
        self.padding_mode = padding_mode
        self.padding_value = padding_value if padding_mode == "constant" else 0

    def forward(self, img):
        if type(img) == PIL.Image.Image:
            img_dim = img.size
        else:
            img_dim = (img.shape[2], img.shape[1])

        padding = [0 if b - a < 0 else math.ceil((b - a)/2)
                   for a, b in zip(img_dim, self.min_img_dim)]
        transform = torchvision.transforms.Pad(
            padding, padding_mode=self.padding_mode, fill=self.padding_value)
        return transform(img)


class Skeleton(th.nn.Module):
    """Applies Canny edge detection on vehicles. It is used for edge-to-car translation."""

    def __init__(self, ratio=0.9, min_length=10, smooth=False, canny=True):
        super().__init__()
        self.ratio = max(0, min(ratio, 1))
        self.min_length = min_length
        self.smooth = smooth
        self.canny = canny

    def forward(self, img):
        if type(img) != PIL.Image.Image:
            raise RuntimeError(
                f"Inputmust be PIL.Image but it is {type(img)}")

        # read the image and convert to gray
        img_ = np.asarray(img)
        gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)

        # enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.canny:
            if self.smooth:
                gray = cv2.medianBlur(gray, 5)

            edge = cv2.Canny(gray, 100, 200)

            # remove short edges
            cnts, _ = cv2.findContours(
                edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts_ = []
            for cnt in cnts:
                if len(cnt) > self.min_length and np.random.uniform(0, 1) < self.ratio:
                    cnts_.append(cnt)

            # re draw the edges
            cnt = cv2.drawContours(
                255*np.ones_like(gray, dtype=np.uint8), cnts_, -1, 0, 1)
        else:
            if self.smooth:
                gray = cv2.GaussianBlur(gray, (5, 5), 3)

            edge = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
            cnt = 255*(edge > 0.1 * np.min(edge)).astype(np.uint8)

        # return PIL.Image
        return PIL.Image.fromarray(cnt).convert('RGB')

#
# Training
#


def compute_loss(predictions: Union[th.Tensor, List[th.Tensor]], label: int, criteria: th.nn.Module):
    """[summary]

    Parameters
    ----------
    predictions : Union[th.Tensor, List[th.Tensor]]
        [description]
    label : int
        [description]
    criteria : th.nn.Module
        [description]

    Returns
    -------
    [type]
        [description]
    """
    loss = 0
    if type(predictions) == list:
        for prediction in predictions:
            gt = label*th.ones_like(prediction,
                                    dtype=prediction.dtype, device=prediction.device)
            loss += criteria(prediction, gt)
    elif type(predictions) == th.Tensor:
        gt = label*th.ones_like(predictions,
                                dtype=predictions.dtype, device=predictions.device)
        loss += criteria(predictions, gt)
    return loss


def weights_init_normal(m):
    """Initialize weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            th.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find("Linear") != -1:
        try:
            th.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find("BatchNorm2d") != -1:
        th.nn.init.normal_(m.weight.data, 1.0, 0.02)
        th.nn.init.constant_(m.bias.data, 0.0)


def weights_init(init_type='gaussian'):
    """https://github.com/NVlabs/MUNIT/blob/bcf8cca2e4934019608a26846526434c5ed86fb5/utils.py"""
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
            try:
                if init_type == 'gaussian':
                    th.nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    th.nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    th.nn.init.kaiming_normal_(
                        m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    th.nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            except:
                pass

    return init_fun


def interpolate(p1: th.Tensor, p2: th.Tensor, ratio: float, use_slerp: bool) -> th.Tensor:
    """Interpolation of two latent points.

    References
    ----------
    https://en.wikipedia.org/wiki/Slerp
    https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
    """

    # linear interpolation if omega -> 0 than it behaves like linear interpolation

    if p1.device != p2.device:
        raise RuntimeError("p1 and p2 uses different devices.")
    if 0 > ratio or ratio > 1:
        raise RuntimeError("Ratio must be between [0, 1]")

    noise = (1 - ratio) * p1 + ratio * p2

    if use_slerp:
        p1_ = p1/(th.linalg.norm(p1) + 1e-10)
        p2_ = p2/(th.linalg.norm(p2) + 1e-10)

        omega = th.acos(th.clip(th.dot(p1_, p2_), -1, 1))
        if not th.isclose(th.sin(omega), th.zeros(1, device=p1.device)):
            noise = (th.sin((1 - ratio)*omega) * p1 +
                     th.sin(ratio*omega) * p2) / th.sin(omega)

    return noise

#
# Evaluation
#


@ th.no_grad()
def vgg16_get_activation_maps(imgs: th.Tensor,
                              layer_idx: int,
                              device: str,
                              normalize_range: Tuple[int, int],
                              global_pooling: bool = True,
                              use_bn: bool = True,
                              img_dim: Tuple[int, int] = (32, 64)) -> th.Tensor:
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
    features = torchvision.models.vgg16(pretrained=True).features.to(
        device) if not use_bn else torchvision.models.vgg16_bn(pretrained=True).features.to(device)
    features.eval()

    # normalize imgs for VGG-16:
    min_, max_ = normalize_range
    imgs_norm = (imgs - min_) / (max_ - min_)
    x = torchvision.transforms.Compose([torchvision.transforms.Resize(img_dim),
                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])(imgs_norm)
    x = x.to(device)
    for i, fet in enumerate(features):
        if i+1 > layer_idx:
            break
        x = fet(x)

    if global_pooling:
        x = th.nn.AdaptiveMaxPool2d(1)(x)

    return x.detach().cpu().clone()


@ th.no_grad()
def perceptual_path_length(generator, n_samples=1024, epsilon=1e-4, use_slerp=True, device="cpu", truncation=1, batch_size=1024, net="vgg"):
    """[summary]

    Parameters
    ----------
    generator : [type]
        [description]
    n_samples : int, optional
        [description], by default 1024
    epsilon : [type], optional
        [description], by default 1e-4
    use_slerp : bool, optional
        [description], by default True
    device : str, optional
        [description], by default "cpu"
    batch_size : int, optional

    Returns
    -------
    [type]
        [description]
    """

    # def slerp(a, b, t):
    #     a = a / a.norm(dim=-1, keepdim=True)
    #     b = b / b.norm(dim=-1, keepdim=True)
    #     d = (a * b).sum(dim=-1, keepdim=True)
    #     p = t * th.acos(d)
    #     c = b - d * a
    #     c = c / c.norm(dim=-1, keepdim=True)
    #     d = a * th.cos(p) + c * th.sin(p)
    #     d = d / d.norm(dim=-1, keepdim=True)
    #     return d

    if n_samples % batch_size != 0:
        raise AttributeError(
            f"Number of samples ({n_samples}) must be divisible by batch size ({batch_size})")

    loss_fn = lpips.LPIPS(net=net).to(device)
    ppl_score = 0

    for _ in range(n_samples // batch_size):
        z1 = th.zeros(size=(batch_size, generator.latent_dim),
                      device=device)
        z2 = th.zeros(size=(batch_size, generator.latent_dim),
                      device=device)
        # point between z1 and z2
        for i in range(batch_size):
            z1_ = th.normal(0, truncation, size=(generator.latent_dim,),
                            device=device)
            z2_ = th.normal(0, truncation, size=(generator.latent_dim,),
                            device=device)

            ratio = np.random.uniform(0, 1)
            # ratio = 1 - 2*epsilon if ratio > 0.5 else 0 # only at the ends
            z1[i, :] = interpolate(z1_, z2_, ratio=ratio,
                                   use_slerp=use_slerp)
            z2[i, :] = interpolate(z1_, z2_, ratio=min(
                ratio + epsilon, 1), use_slerp=use_slerp)

        # create images
        imgs1 = generator(z1)
        imgs2 = generator(z2)

        # compute lpips
        ppl_score += th.sum(loss_fn(imgs1, imgs2,
                            normalize=True)).item() / n_samples

    # return mean lpips
    return ppl_score / epsilon**2


def fid_score(imgs1: th.Tensor,
              imgs2: th.Tensor,
              n_cases: int = 1024,
              layer_idx: int = 33,
              device: str = "cpu",
              normalize_range: Tuple[int, int] = (-1, 1),
              use_bn: bool = True,
              skip_vgg: bool = False,
              memorized_fid: bool = False) -> float:
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

    if not skip_vgg:
        # activation of dataset 1
        act1 = np.squeeze(vgg16_get_activation_maps(
            imgs1[:n_cases], layer_idx, device, normalize_range, use_bn=use_bn).numpy())

        # activation of dataset 2
        act2 = np.squeeze(vgg16_get_activation_maps(
            imgs2[:n_cases], layer_idx, device, normalize_range, use_bn=use_bn).numpy())
    else:
        act1 = imgs1[:n_cases]
        if type(imgs1) == th.Tensor:
            act1 = np.squeeze(imgs1[:n_cases].detach().cpu().numpy())

        act2 = imgs2[:n_cases]
        if type(imgs2) == th.Tensor:
            act2 = np.squeeze(imgs2[:n_cases].detach().cpu().numpy())

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

    if memorized_fid:
        # act2 is generated
        # act1 is real images

        # norms inverse, replace inf values with 0
        act1_norm = np.expand_dims(np.linalg.norm(act1, axis=1), 1)
        act2_norm = np.expand_dims(np.linalg.norm(act2, axis=1), 1)
        norms_inverse = 1 / (act2_norm.dot(act1_norm.T))

        norms_inverse[np.isinf(norms_inverse)] = 0

        # cos distance between two activations
        cos_distances = (1 - norms_inverse*act2.dot(act1.T))
        cos_distance = cos_distances.min(axis=1)  # min for each row
        d = np.mean(cos_distance)

        import matplotlib.pyplot as plt
        plt.hist(cos_distance)
        plt.show()

        fid = fid / (d if d > 1e-10 else 1)
    return fid


if __name__ == "__main__":
    z1_ = th.normal(0, 1, size=(100,),
                    device="cpu")
    z2_ = th.normal(0, 1, size=(100,),
                    device="cpu")

    ratio = np.random.uniform(0, 1)
    # z1 = slerp(z1_, z2_, t=ratio)
    # z2 = slerp(z1_, z2_, t=1 - ratio)
    z1 = interpolate(z1_, z2_, ratio, True)
    z2 = interpolate(z2_, z1_, 1 - ratio, True)
    print(th.allclose(z1, z2))
