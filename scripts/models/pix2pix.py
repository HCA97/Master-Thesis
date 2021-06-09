import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from ..layers import *
from ..utility import *
from .networks import *

# -------------------------------------------------------- #
#                         GANS                             #
# -------------------------------------------------------- #


class GAN(pl.LightningModule):
    """My Gan module because I find ``pl_bolts`` implementation confusing.
    It is based on https://lightning-bolts.readthedocs.io/en/latest/gans.html#basic-gan

    Parameters
    ----------

    """

    def __init__(self,
                 img_dim=(3, 32, 64),
                 learning_rate=0.0002,
                 latent_dim=100,
                 betas=(0.5, 0.999),
                 discriminator_params=None,
                 generator_params=None,
                 use_gp=False,
                 alpha=1.0,
                 beta=1.0,
                 gen_model="basic",
                 disc_model="basic"):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = self.get_criterion()

    def get_criterion(self):
        return nn.BCELoss()

    def get_generator(self, generator_params):
        generator_params = generator_params if generator_params else {}

        if self.hparams.gen_model == "resnet":
            generator = ResNetGenerator(self.hparams.img_dim,
                                        self.hparams.latent_dim, **generator_params)
        elif self.hparams.gen_model == "basic":
            generator = BasicGenerator(self.hparams.img_dim,
                                       self.hparams.latent_dim, **generator_params)
            generator.apply(weights_init_normal)
        elif self.hparams.gen_model == "unet":
            generator = UnetGenerator(
                n_channels=self.hparams.img_dim[0], **generator_params)
        else:
            raise NotImplementedError()

        return generator

    def get_discriminator(self, discriminator_params):
        discriminator_params = discriminator_params if discriminator_params else {}

        if self.hparams.disc_model == "basic":
            discriminator = BasicDiscriminator(
                self.hparams.img_dim, **discriminator_params)
            discriminator.apply(weights_init_normal)
        else:
            raise NotImplementedError()

        return discriminator

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = self.hparams.betas
        opt_disc = th.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = th.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        return self.generator(tensor)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, fake = batch

        result = None

        # Train discriminator
        if optimizer_idx == 0:
            if self.hparams.use_gp:
                real = real.requires_grad_(True)

            # Train with real
            real_pred = self.discriminator(real)
            real_gt = th.ones_like(real_pred)
            real_loss = self.criterion(real_pred, real_gt)

            # Gradient Penalty (R1)
            gp_loss = 0
            if self.hparams.use_gp:
                grad = th.autograd.grad(real_pred, real, grad_outputs=th.ones(real_pred.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                slopes = th.sqrt(th.sum(th.mul(grad, grad), dim=[1, 2, 3]))
                gp_loss = 0.5 * self.hparams.alpha * th.mean(slopes)

            # Train with fake
            if self.hparams.gen_model == "unet":
                fake_ = fake + self.generator(fake)
            else:
                z = th.normal(0, 1, size=(len(real), self.hparams.latent_dim))
                fake_ = self.generator(z)
            fake_pred = self.discriminator(fake_)
            fake_gt = th.zeros_like(fake_pred)
            fake_loss = self.criterion(fake_pred, fake_gt)

            # Total Loss
            result = real_loss + fake_loss + gp_loss

            # Logging
            self.log("loss/disc", result)
            self.log("loss/disc_real", real_loss)
            self.log("loss/disc_fake", fake_loss)
            if self.hparams.use_gp:
                self.log("loss/disc_gp", gp_loss)

        # Train generator
        if optimizer_idx == 1:
            l1_loss = 0
            if self.hparams.gen_model == "unet":
                delta = self.generator(fake)
                fake_ = fake + delta

                # L1-Norm
                l1_loss = self.hparams.beta * th.sum(th.abs(delta))
            else:
                z = th.normal(0, 1, size=(len(real), self.hparams.latent_dim))
                fake_ = self.generator(z)

            # Gen Loss
            fake_pred = self.discriminator(fake_)
            fake_gt = th.ones_like(fake_pred)
            gen_loss = self.criterion(fake_pred, fake_gt)

            # Total Loss
            result = gen_loss + l1_loss

            # Logging
            self.log("loss/gen", result)
            if self.hparams.gen_model == "unet":
                self.log("loss/gen_l1", l1_loss)

        return result
