import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


from scripts.models import *
from scripts.utility import *

# from pl_bolts.models.gans import DCGAN
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pytorch_lightning import loggers as pl_loggers


class MNISTDataModule(pl.LightningModule):
    def __init__(self, img_size=(32, 32), data_dir="./data/mnist", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = None

    def setup(self, state=None):
        self.has_setup_fit = True
        self.dataset = datasets.MNIST(self.data_dir,
                                      train=True, download=True,
                                      transform=transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                                                    transforms.Normalize([0.5], [0.5])]))

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader


class GAN(pl.LightningModule):
    """My Gan module because I find `pl_bolt` implementation confusing.

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
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input noise
        Example::
            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # print(self.logger.experiment)
        print(self.log)
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def dics_step(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        # Logging
        self.log("loss/disc", disc_loss, on_epoch=True)
        self.log("loss/disc_real", real_loss, on_epoch=True)
        self.log("loss/disc_fake", fake_loss, on_epoch=True)

        return disc_loss

    def _disc_step(self, real: torch.Tensor) -> torch.Tensor:
        disc_loss = self._get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)

        return gen_loss

    def _get_fake_pred(self, real: torch.Tensor) -> torch.Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    # @staticmethod
    # def add_model_specific_args(parent_parser: ) -> ArgumentParser:
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument("--beta1", default=0.5, type=float)
    #     parser.add_argument("--feature_maps_gen", default=64, type=int)
    #     parser.add_argument("--feature_maps_disc", default=64, type=int)
    #     parser.add_argument("--latent_dim", default=100, type=int)
    #     parser.add_argument("--learning_rate", default=0.0002, type=float)
    #     return parser


model = GAN((1, 32, 32))
mnist = MNISTDataModule()
mnist.setup()

tb_logger = pl_loggers.TensorBoardLogger('logs/')

callbacks = [
    TensorboardGenerativeModelImageSampler(num_samples=16),
    LatentDimInterpolator(interpolate_epoch_interval=1),

]
trainer = pl.Trainer(logger=tb_logger, gpus=1,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=mnist)
