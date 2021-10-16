import copy
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lpips
import numpy as np

from ..layers import *
from ..utility import *
from .networks import *

# -------------------------------------------------------- #
#                         GANS                             #
# -------------------------------------------------------- #


class GAN(pl.LightningModule):
    """

    Parameters
    ----------

    """

    def __init__(self,
                 img_dim=(3, 32, 64),
                 fid_interval=5,
                 # For optimizer
                 use_lr_scheduler=False,
                 learning_rate_gen=0.0002,
                 learning_rate_disc=0.0002,
                 weight_decay_disc=0.0,
                 weight_decay_gen=0.0,
                 betas=(0.5, 0.999),
                 # Model Parameters
                 discriminator_params=None,
                 generator_params=None,
                 gen_model="basic",
                 disc_model="basic",
                 gen_init="normal",
                 disc_init="normal",
                 # GAN Loss
                 use_symmetry_loss=False,
                 gamma=1,
                 **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = self.get_criterion()

        # for FID score computation
        self.n_samples = 1024
        self.act_real = []
        self.act_samples = 64
        self.gen_input = []
        self.imgs_real = None

    def lr_schedulers(self):
        # over-write this shit
        return [self.scheduler_disc, self.scheduler_gen]

    def get_criterion(self):
        return nn.BCELoss()

    def get_generator(self, generator_params):
        generator_params = generator_params if generator_params else {}

        if self.hparams.gen_model == "basic":
            generator = BasicGenerator(
                self.hparams.img_dim, **generator_params)
        elif self.hparams.gen_model == "resnet":
            generator = ResNetGenerator(
                self.hparams.img_dim, **generator_params)
        else:
            raise NotImplementedError()

        if self.hparams.gen_init == "normal":
            generator.apply(weights_init_normal)
        return generator

    def get_discriminator(self, discriminator_params):
        discriminator_params = discriminator_params if discriminator_params else {}

        if self.hparams.disc_model == "basic":
            discriminator = BasicDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        elif self.hparams.disc_model == "patch":
            discriminator = PatchDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        elif self.hparams.disc_model == "basicpatch":
            discriminator = BasicPatchDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        elif self.hparams.disc_model == "resnet":
            discriminator = ResNetDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        else:
            raise NotImplementedError()

        if self.hparams.disc_init == "normal":
            discriminator.apply(weights_init_normal)
        return discriminator

    def configure_optimizers(self):

        # optimizers
        opt_disc = th.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate_disc,
            weight_decay=weight_decay_disc,
            betas=self.hparams.betas)
        opt_gen = th.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate_gen,
            weight_decay=weight_decay_gen,
            betas=self.hparams.betas)

        # scheduler - patience is 125 epochs
        # pytorch lighting is annoying
        if self.hparams.fid_interval > 0:
            patience = 125 // self.hparams.fid_interval
            self.scheduler_disc = ReduceLROnPlateau(
                opt_disc, 'min', patience=patience, factor=0.5, verbose=True)
            self.scheduler_gen = ReduceLROnPlateau(
                opt_gen, 'min', patience=patience, factor=0.5, verbose=True)
        return [opt_disc, opt_gen], []

    def forward(self, tensor: th.Tensor, **kwargs) -> th.Tensor:
        return self.generator(tensor)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, fake = batch

        if hparams_initial.fid_interval > 0:
            # append real image activation for FID computation
            if len(self.act_real) < self.n_samples:
                act = np.squeeze(vgg16_get_activation_maps(
                    real, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
                self.act_real = np.concatenate(
                    (self.act_real, act), axis=0) if len(self.act_real) > 0 else act

            # append fake images or noise vectors for FID computation
            if len(self.gen_input) < self.n_samples:
                self.gen_input = th.normal(
                    0, 1, (self.n_samples, self.generator.latent_dim), device=self.device)

        result = None

        # Train discriminator
        if optimizer_idx == 0:

            # Loss
            real_pred = self.discriminator(real)
            real_loss = compute_loss(real_pred, 1, self.criterion)

            # Train with fake
            z = th.normal(0, 1, size=(
                len(real), self.generator.latent_dim), device=self.device)
            fake_ = self.generator(z)

            # Loss
            fake_pred = self.discriminator(fake_)
            fake_loss = compute_loss(fake_pred, 0, self.criterion)

            # Total Loss
            result = real_loss + fake_loss

            # Logging
            self.log("loss/disc", result)
            self.log("loss/disc_real", real_loss)
            self.log("loss/disc_fake", fake_loss)

        # Train generator
        if optimizer_idx == 1:
            z = th.normal(0, 1, size=(
                len(real), self.generator.latent_dim), device=self.device)
            fake_ = self.generator(z)

            symmetry_loss = 0
            # car must be symmetical
            if self.hparams.use_symmetry_loss:
                r = self.hparams.img_dim[1]
                idx_1 = list(range(0, r//2))
                idx_2 = list(range(r-1, r//2-1, -1))

                fake_1 = fake_[:, :, idx_1, :]
                fake_2 = fake_[:, :, idx_2, :]

                symmetry_loss = self.hparams.gamma * \
                    th.mean(th.abs(fake_1 - fake_2))

            # Gen Loss
            fake_pred = self.discriminator(fake_)
            gen_loss = compute_loss(fake_pred, 1, self.criterion)

            # Total Loss
            result = gen_loss + symmetry_loss

            # Logging
            self.log("loss/gen", gen_loss)
            if self.hparams.use_symmetry_loss:
                self.log("loss/gen_symmetry", symmetry_loss)

        return result

    def on_epoch_end(self):

        if ((self.current_epoch + 1) % self.hparams.fid_interval == 0 or self.current_epoch == 0) and \
                len(self.gen_input) >= self.n_samples and self.hparams.fid_interval > 0:

            self.generator.eval()

            with th.no_grad():
                act_fake = np.zeros_like(self.act_real)

                for i in range(self.n_samples // self.act_samples):
                    # start and end index
                    si = i * self.act_samples
                    ei = min((i+1)*self.act_samples, self.n_samples)

                    # images generated by generator
                    imgs = self.generator(self.gen_input[si:ei])

                    # vgg activations
                    act = np.squeeze(vgg16_get_activation_maps(
                        imgs, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
                    act_fake[si:ei] = act

            self.generator.train()

            # compute fid
            fid = fid_score(self.act_real, act_fake, skip_vgg=True)

            # log fid
            self.log("fid", fid)

            # lr scheduler
            # is this correct place to update?
            if self.hparams.use_lr_scheduler:
                sch1, sch2 = self.lr_schedulers()
                sch1.step(fid)
                sch2.step(fid)

        # log lr
        opt1, opt2 = self.optimizers(use_pl_optimizer=True)
        self.log("lr/disc", opt1.param_groups[0]["lr"])
        self.log("lr/gen", opt2.param_groups[0]["lr"])
