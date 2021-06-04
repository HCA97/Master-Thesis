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

    def __init__(self, img_dim, learning_rate=0.0002, latent_dim=100, betas=(0.5, 0.999), discriminator_params=None, generator_params=None, use_gp=False, alpha=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = self.get_criterion()

    def get_criterion(self):
        return nn.BCELoss()

    def get_generator(self, generator_params):
        generator_params = generator_params if generator_params else {}
        generator = BasicGenerator(
            self.hparams.img_dim, self.hparams.latent_dim, **generator_params)
        generator.apply(weights_init_normal)
        return generator

    def get_discriminator(self, discriminator_params):
        discriminator_params = discriminator_params if discriminator_params else {}
        discriminator = BasicDiscriminator(
            self.hparams.img_dim, **discriminator_params)
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
        real = real.requires_grad_(True)
        real_pred = self.discriminator(real)
        real_gt = th.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Gradient Penalty (R1)
        gp_loss = 0
        if self.hparams.use_gp:
            # https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/Losses.py#L174
            grad = th.autograd.grad(real_pred, real, grad_outputs=th.ones(real_pred.size()).to(self.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0].view(real.size(0), -1)
            gp_loss = self.hparams.alpha * th.sum(th.mul(grad, grad))
            # print(gp_loss)

        # Train with fake
        fake_pred = self.get_pred_fake(real)
        fake_gt = th.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        # Total Loss
        disc_loss = real_loss + fake_loss + gp_loss

        # Logging
        self.log("loss/disc", disc_loss)
        self.log("loss/disc_real", real_loss)
        self.log("loss/disc_fake", fake_loss)
        if self.hparams.use_gp:
            self.log("loss/disc_gp", gp_loss)

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
