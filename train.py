# from argparse import ArgumentParser

from torch.nn import BCELoss
import torch as th
# from torchsummary import summary
import pytorch_lightning as pl

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *


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
        return BCELoss()

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

        # save weights
        writer = self.logger.experiment

        with th.no_grad():
            for name, params in self.generator.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(
                        "Generator/" + name, params.detach().cpu().numpy(), self.global_step)
            for name, params in self.discriminator.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(
                        "Discriminator/" + name, params.detach().cpu().numpy(), self.global_step)

    # @staticmethod
    # def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument("--beta1", default=0.5, type=float)
    #     parser.add_argument("--beta2", default=0.999, type=float)
    #     parser.add_argument("--n_channels", default=1, type=int)
    #     parser.add_argument("--img_width", default=32, type=int)
    #     parser.add_argument("--img_dim", default=(1, 32, 32), type=int)
    #     parser.add_argument("--feature_maps_disc", default=64, type=int)
    #     parser.add_argument("--latent_dim", default=100, type=int)
    #     parser.add_argument("--learning_rate", default=0.0002, type=float)
    #     return parser


# MNIST TEST EXAMPLE
# model = GAN((1, 32, 32))
# mnist = MNISTDataModule(batch_size=128)
# mnist.setup()

# callbacks = [
#     TensorboardGeneratorSampler(num_samples=64),
#     LatentDimInterpolator(interpolate_epoch_interval=1)]

# # Apparently Trainer has logger by default
# trainer = pl.Trainer(default_root_dir="logs", gpus=1, max_epochs=1,
#                      callbacks=callbacks, progress_bar_refresh_rate=20)
# trainer.fit(model, datamodule=mnist)


# POTSDAM CARS
model = GAN((3, 32, 64))
# print(summary(model.discriminator.cuda(), (3, 32, 64)))
# print(summary(model.generator.cuda(), (100, 1, 1)))
potsdam = PostdamCarsDataModule("../potsdam_data/potsdam_cars", batch_size=64)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(num_samples=64),
    LatentDimInterpolator(interpolate_epoch_interval=1)]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir="logs", gpus=1, max_epochs=5,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=potsdam)
