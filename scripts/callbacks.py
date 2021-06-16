from typing import List, Optional, Tuple

import torch as th
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor

from .utility import interpolate, fid_score, vgg16_get_activation_maps

_TORCHVISION_AVAILABLE = True
try:
    import torchvision
except ImportError:
    _TORCHVISION_AVAILABLE = False


class MyEarlyStopping(Callback):
    def __init__(self, epoch=300, threshold=5, monitor="fid", mode="min"):
        super().__init__()
        self.threshold = threshold
        self.epoch = epoch
        self.monitor = monitor
        self.mode = np.less if mode == "min" else np.greater

    def on_epoch_end(self, trainer, pl_module):
        monitor_val = trainer.callback_metrics.get(self.monitor)
        if monitor_val:
            if (trainer.current_epoch + 1) >= self.epoch and self.mode(self.threshold, monitor_val.item()):
                raise KeyboardInterrupt("Interrupted by my early stopping")


class Pix2PixCallback(Callback):
    def __init__(self, epoch_interval=1, n_samples=5, normalize=True):
        super().__init__()

        self.epoch_interval = epoch_interval
        self.n_samples = n_samples
        self.normalize = normalize

    def on_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch + 1) % self.epoch_interval == 0 or trainer.current_epoch == 0:
            if pl_module.hparams.gen_model in ["unet", "refiner"]:
                pl_module.eval()

                n_gen_input = min(len(pl_module.gen_input), self.n_samples)
                n_imgs_real = min(len(pl_module.imgs_real), self.n_samples)

                # pick n_samples fake cases
                gen_input = pl_module.gen_input[:n_gen_input]
                imgs_fake = pl_module(gen_input)

                # make a grid
                imgs = th.cat(
                    [gen_input, imgs_fake, imgs_fake - gen_input], dim=0)
                grid = torchvision.utils.make_grid(
                    imgs, nrow=n_gen_input, normalize=self.normalize)

                # logging
                str_title = f'{pl_module.__class__.__name__}_gen_fake_imgs'
                trainer.logger.experiment.add_image(
                    str_title, grid, global_step=trainer.current_epoch)

                # pick n_samples real cases
                gen_input = pl_module.imgs_real[:n_imgs_real]
                imgs_fake = pl_module(gen_input)

                # make a grid
                imgs = th.cat(
                    [gen_input, imgs_fake, imgs_fake - gen_input], dim=0)
                grid = torchvision.utils.make_grid(
                    imgs, nrow=n_imgs_real, normalize=self.normalize)

                # logging
                str_title = f'{pl_module.__class__.__name__}_gen_real_imgs'
                trainer.logger.experiment.add_image(
                    str_title, grid, global_step=trainer.current_epoch)

                pl_module.train()


class ShowWeights(Callback):
    def __init__(self, epoch_interval=1):
        super().__init__()
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch + 1) % self.epoch_interval == 0 or trainer.current_epoch == 0:
            writer = trainer.logger.experiment

            pl_module.eval()
            with th.no_grad():
                if pl_module.hparams.moving_average:
                    for (name, params), params_avg in zip(pl_module.generator.named_parameters(), pl_module.generator_avg.parameters()):
                        if params.grad is not None:
                            writer.add_histogram(
                                "Generator/" + name, params.detach().cpu().numpy(), trainer.current_epoch)
                            writer.add_histogram(
                                "Generator_Avg/" + name, params_avg.detach().cpu().numpy(), trainer.current_epoch)
                else:
                    for name, params in pl_module.generator.named_parameters():
                        if params.grad is not None:
                            writer.add_histogram(
                                "Generator/" + name, params.detach().cpu().numpy(), trainer.current_epoch)

                for name, params in pl_module.discriminator.named_parameters():
                    if params.grad is not None:
                        writer.add_histogram(
                            "Discriminator/" + name, params.detach().cpu().numpy(), trainer.current_epoch)
            pl_module.train()


class LatentDimInterpolator(Callback):
    """Similar implementation of
    https://lightning-bolts.readthedocs.io/en/latest/variational_callbacks.html#

    I just didn't wanted to include new library for two callbacks functions.
    """

    def __init__(self, interpolate_epoch_interval: int = 1,
                 num_samples: int = 5, steps: int = 10, normalize: bool = True, use_slerp: bool = True):
        super().__init__()

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `torchvision` which is not installed yet.")

        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps
        self.use_slerp = use_slerp
        self.init_noise = False

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0 or trainer.current_epoch == 0:
            if pl_module.hparams.gen_model not in ["unet", "refiner"]:
                images = self.interpolate_latent_space(pl_module)
                # images = th.cat(images, dim=0)  # type: ignore[assignment]

                num_rows = self.steps + 2
                grid = torchvision.utils.make_grid(
                    images, nrow=num_rows, normalize=self.normalize)
                str_title = f'{pl_module.__class__.__name__}_latent_space'
                trainer.logger.experiment.add_image(
                    str_title, grid, global_step=trainer.current_epoch)

    def interpolate_latent_space(self, pl_module: LightningModule, truncation: float = 1) -> List[Tensor]:
        images = []
        latent_dim = pl_module.generator.latent_dim

        with th.no_grad():
            pl_module.eval()

            if not self.init_noise:
                self.init_noise = True
                self.z1 = th.normal(
                    mean=0, std=truncation, size=(self.num_samples, latent_dim), device=pl_module.device)
                self.z2 = th.normal(mean=0, std=truncation, size=(
                    self.num_samples, latent_dim), device=pl_module.device)

            z1 = self.z1.clone().detach()
            z2 = self.z2.clone().detach()

            ratios = th.linspace(0, 1, steps=self.steps)

            for i in range(self.num_samples):
                # interpolate points
                points = th.zeros(
                    (self.steps + 2, latent_dim), device=pl_module.device)

                points[0] = z1[i]
                points[-1] = z2[i]

                for j in range(self.steps):
                    points[j+1] = interpolate(z1[i],
                                              z2[i], ratios[j], self.use_slerp)

                # generate images
                fake_imgs = pl_module(points)

                # append images
                for img in fake_imgs:
                    images.append(img)

        pl_module.train()
        return images


class TensorboardGeneratorSampler(Callback):
    """
    Generates images and logs to tensorboard.

    References
    ----------
    https://github.com/PyTorchLightning/lightning-bolts/blob/c3b60de7dc30c5f7947256479d9be3a042b8c182/pl_bolts/callbacks/vision/image_generation.py#L15
    """

    def __init__(
        self,
        epoch_interval: int = 5,
        num_samples: int = 18,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.epoch_interval = epoch_interval
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.z = None

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if (trainer.current_epoch + 1) % self.epoch_interval == 0 or trainer.current_epoch == 0:
            if pl_module.hparams.gen_model not in ["unet", "refiner"]:
                dim = (self.num_samples, pl_module.generator.latent_dim, 1, 1)
                if self.z is None:
                    self.z = th.normal(mean=0.0, std=1.0, size=dim,
                                       device=pl_module.device)

                # generate images
                with th.no_grad():
                    pl_module.eval()
                    images = pl_module(self.z)
                    pl_module.train()

                grid = torchvision.utils.make_grid(
                    tensor=images,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )
                str_title = f"{pl_module.__class__.__name__}_images"
                trainer.logger.experiment.add_image(
                    str_title, grid, global_step=trainer.current_epoch)
