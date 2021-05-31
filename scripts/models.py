import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from .layers import *
from .utility import *

# -------------------------------------------------------- #
#                   Discriminators                         #
# -------------------------------------------------------- #


class BasicDiscriminator(nn.Module):
    """Basic Discriminator. It is similar to the
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

    Parameters
    ----------
    img_size : list or tuple
        Input image size (input_channels, input_height, input_width)

    References
    ----------
    Alec Radford and Luke Met and Soumith Chintala. 2016.
    UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS.
    """

    def __init__(self, img_size):
        super().__init__()

        input_channels, input_height, input_width = img_size
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_channels, 16, stride=2,
                      dropout=True, use_bn=False),
            ConvBlock(16, 32, stride=2, dropout=True),
            ConvBlock(32, 64, stride=2, dropout=True),
            ConvBlock(64, 128, stride=2, dropout=True)
        )

        self.ds_size = (input_width // 2**4) * (input_height // 2**4)
        self.l1 = nn.Sequential(
            nn.Linear(128 * self.ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        x = self.conv_blocks(img)
        x = x.view(x.shape[0], 128 * self.ds_size)
        x = self.l1(x)
        return x

# -------------------------------------------------------- #
#                       Generators                         #
# -------------------------------------------------------- #


class BasicGenerator(nn.Module):
    """Basic Generator. It is similar to the 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

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

    def __init__(self, img_size, latent_dim=100, init_channels=128):
        super().__init__()

        input_channels, input_height, input_width = img_size
        self.init_height = input_height // 4
        self.init_width = input_width // 4
        self.latent_dim = latent_dim
        self.init_channels = init_channels

        self.l1 = nn.Linear(self.latent_dim, self.init_channels *
                            self.init_width * self.init_height)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_channels),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            ConvBlock(64, input_channels, use_bn=False, act="tanh")
        )

    def forward(self, z):
        x = z.view(z.shape[0], self.latent_dim)
        x = self.l1(x)
        x = x.view(x.shape[0], self.init_channels,
                   self.init_height, self.init_width)
        x = self.conv_blocks(x)
        return x

# -------------------------------------------------------- #
#                         GANS                             #
# -------------------------------------------------------- #


class GAN(pl.LightningModule):
    """My Gan module because I find ``pl_bolts`` implementation confusing.
    It is based on https://lightning-bolts.readthedocs.io/en/latest/gans.html#basic-gan

    Parameters
    ----------

    """

    def __init__(self, img_dim, learning_rate=0.0002, latent_dim=100, betas=(0.5, 0.999)):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(img_dim)
        self.discriminator = self.get_discriminator(img_dim)
        self.criterion = self.get_criterion()

    def get_criterion(self):
        return nn.BCELoss()

    def get_generator(self, img_dim):
        generator = BasicGenerator(img_dim, self.hparams.latent_dim)
        generator.apply(weights_init_normal)
        return generator

    def get_discriminator(self, img_dim):
        discriminator = BasicDiscriminator(img_dim)
        discriminator.apply(weights_init_normal)
        return discriminator

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = self.hparams.betas
        opt_disc = th.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = th.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: th.Tensor) -> th.Tensor:
        """
        Generates an image given input noise

        Example::
            noise = th.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self.disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self.gen_step(real)

        return result

    def disc_step(self, real: th.Tensor) -> th.Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = th.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self.get_pred_fake(real)
        fake_gt = th.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        # Total Loss
        disc_loss = real_loss + fake_loss

        # Logging
        self.log("loss/disc", disc_loss)
        self.log("loss/disc_real", real_loss)
        self.log("loss/disc_fake", fake_loss)

        return disc_loss

    def get_pred_fake(self, real: th.Tensor) -> th.Tensor:
        noise = th.normal(mean=0.0, std=1.0, size=(
            len(real), self.hparams.latent_dim, 1, 1), device=self.device)
        fake = self.generator(noise)
        return self.discriminator(fake)

    def gen_step(self, real: th.Tensor) -> th.Tensor:

        fake_pred = self.get_pred_fake(real)
        fake_gt = th.ones_like(fake_pred)

        gen_loss = self.criterion(fake_pred, fake_gt)

        self.log("loss/gen", gen_loss)
        return gen_loss

    def on_epoch_end(self):

        writer = self.logger.experiment

        self.generator.eval()
        self.discriminator.eval()

        with th.no_grad():
            # save weights
            for name, params in self.generator.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(
                        "Generator/" + name, params.detach().cpu().numpy(), self.global_step)
            for name, params in self.discriminator.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(
                        "Discriminator/" + name, params.detach().cpu().numpy(), self.global_step)

            # save activations

        self.generator.train()
        self.discriminator.train()
