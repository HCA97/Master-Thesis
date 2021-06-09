from typing import List, Optional, Tuple

import torch as th
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor

_TORCHVISION_AVAILABLE = True
try:
    import torchvision
except ImportError:
    _TORCHVISION_AVAILABLE = False


class SaveWeights(Callback):
    def __init__(self, epoch_interval=1):
        super().__init__()
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            writer = trainer.logger.experiment

            pl_module.eval()
            with th.no_grad():
                # save weights
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

        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:

            images = self.interpolate_latent_space(pl_module)
            # images = th.cat(images, dim=0)  # type: ignore[assignment]

            num_rows = self.steps + 2
            grid = torchvision.utils.make_grid(
                images, nrow=num_rows, normalize=self.normalize)
            str_title = f'{pl_module.__class__.__name__}_latent_space'
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.current_epoch)

    def interpolate_latent_space(self, pl_module: LightningModule) -> List[Tensor]:
        images = []
        latent_dim = pl_module.hparams.latent_dim

        with th.no_grad():
            pl_module.eval()

            if not self.init_noise:
                self.init_noise = True
                self.z1 = th.normal(
                    mean=0, std=1, size=(self.num_samples, latent_dim), device=pl_module.device)
                self.z2 = th.normal(mean=0, std=1, size=(
                    self.num_samples, latent_dim), device=pl_module.device)

            z1 = self.z1.clone().detach()
            z2 = self.z2.clone().detach()

            for i in range(self.num_samples):
                # interpolate points
                points = self.interpolate(z1[i], z2[i], pl_module)
                points = points.view(points.shape[0], latent_dim, 1, 1)

                # generate images
                fake_imgs = pl_module(points)

                # append images
                for img in fake_imgs:
                    images.append(img)

        pl_module.train()
        return images

    def interpolate(self, p1: Tensor, p2: Tensor, pl_module: LightningModule) -> Tensor:
        """Interpolation of two latent points.

        References
        ----------
        https://en.wikipedia.org/wiki/Slerp
        https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
        """

        points = th.zeros((self.steps + 2, len(p1)), device=pl_module.device)
        ratios = th.linspace(0, 1, steps=self.steps)

        points[0] = p1
        points[-1] = p2

        for i, ratio in enumerate(ratios):
            # linear interpolation if omega -> 0 than it behaves like linear interpolation
            noise = (1 - ratio) * p1 + ratio * p2

            if self.use_slerp:
                p1_ = p1/(th.linalg.norm(p1) + 1e-10)
                p2_ = p2/(th.linalg.norm(p2) + 1e-10)

                omega = th.acos(th.clip(th.dot(p1_, p2_), -1, 1))
                if not th.isclose(th.sin(omega), th.zeros(1, device=pl_module.device)):
                    noise = (th.sin((1 - ratio)*omega) * p1 +
                             th.sin(ratio*omega) * p2) / th.sin(omega)

            points[i+1] = noise
        return points


class TensorboardGeneratorSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = th.rand(batch_size, latent_dim)
        img_samples = your_model(z)
    Example::
        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])

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
        # type: ignore[union-attr]

        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            dim = (self.num_samples, pl_module.hparams.latent_dim, 1, 1)
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
