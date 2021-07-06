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


class MUNIT(pl.LightningModule):
    """My Gan module because I find ``pl_bolts`` implementation confusing.
    It is based on https://lightning-bolts.readthedocs.io/en/latest/gans.html#basic-gan

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
                 betas=(0.5, 0.999),
                 # Model Parameters
                 discriminator_params=None,
                 generator_params=None,
                 gen_model="munit",
                 disc_model="munit",
                 gen_init="normal",
                 disc_init="normal",
                 # Training Loss
                 criterion="hinge",
                 use_symmetry_loss=False,
                 gamma=1,
                 use_lpips=False,
                 alpha=1,
                 # Generator Loss
                 l0=10,
                 l1=1,
                 l2=1,
                 l3=1,
                 l4=1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = L2Nrom() if criterion == "hinge" else L2Norm

        if rec_loss == "lpips":
            self.lpips = lpips.LPIPS(net='alexnet')

        # ingerating FID score
        self.n_samples = 1024
        self.act_real = []
        self.act_samples = 64
        self.gen_input = []
        self.imgs_real = None

        # LR Scheduler
        # self.scheduler_disc, self.scheduler_gen = None, None

    def lr_schedulers(self):
        # over-write this shit
        return [self.scheduler_disc, self.scheduler_gen]

    def get_criterion(self):
        return nn.BCELoss()

    def get_generator(self, generator_params):
        generator_params = generator_params if generator_params else {}

        if self.hparams.gen_model == "resnet":
            generator = ResNetGenerator(
                self.hparams.img_dim, **generator_params)
        elif self.hparams.gen_model == "basic":
            generator = BasicGenerator(
                self.hparams.img_dim, **generator_params)
        elif self.hparams.gen_model == "unet":
            generator = UnetGenerator(
                n_channels=self.hparams.img_dim[0], **generator_params)
        elif self.hparams.gen_model == "refiner":
            generator = RefinerNet(
                n_channels=self.hparams.img_dim[0], **generator_params)
        elif self.hparams.gen_model == "stylegan":
            generator = StyleGan(self.hparams.img_dim, **generator_params)
        else:
            raise NotImplementedError()

        if self.hparams.gen_init == "normal":
            generator.apply(weights_init_normal)
        if self.hparams.gen_init == "stylegan":
            generator.apply(weights_init_stylegan)
        return generator

    def get_discriminator(self, discriminator_params):
        discriminator_params = discriminator_params if discriminator_params else {}

        if self.hparams.disc_model == "basic":
            discriminator = BasicDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        elif self.hparams.disc_model == "patch":
            discriminator = PatchDiscriminator(
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
            betas=self.hparams.betas)
        opt_gen = th.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate_gen,
            betas=self.hparams.betas)

        # scheduler - patience is 125 epochs
        # pytorch lighting is annoying
        patience = 125 // self.hparams.fid_interval
        self.scheduler_disc = ReduceLROnPlateau(
            opt_disc, 'min', patience=patience, factor=0.5, verbose=True)
        self.scheduler_gen = ReduceLROnPlateau(
            opt_gen, 'min', patience=patience, factor=0.5, verbose=True)
        return [opt_disc, opt_gen], []

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        if self.hparams.moving_average:
            return self.generator_avg(tensor)
        return self.generator(tensor)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, fake = batch

        # append real image activation for FID
        if len(self.act_real) < self.n_samples:
            act = np.squeeze(vgg16_get_activation_maps(
                real, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
            self.act_real = np.concatenate(
                (self.act_real, act), axis=0) if len(self.act_real) > 0 else act

        # append fake images or noise vectors for FID
        if len(self.gen_input) < self.n_samples:
            if self.hparams.gen_model in ["unet", "refiner"]:
                self.gen_input = th.cat((self.gen_input, fake), dim=0) if len(
                    self.gen_input) > 0 else fake
            else:
                self.gen_input = th.normal(
                    0, 1, (self.n_samples, self.generator.latent_dim), device=self.device)

        # store the first batch for callback
        if self.imgs_real is None:
            self.imgs_real = real

        result = None

        # DISCRIMINATOR
        if optimizer_idx == 0:

            # Real Loss
            label = 1
            if self.hparams.one_sided_label_smoothing:
                label = np.random.uniform(0.8, 1)

            real_ret1 = self.discriminator(fake, 0)
            real_loss1 = compute_loss(real_ret1, label, self.criterion)

            real_ret2 = self.discriminator(real, 1)
            real_loss2 = compute_loss(real_ret2, label, self.criterion)

            real_loss = real_loss1 + real_loss2

            # Fake Loss
            ret = self.generator(fake, real)
            x12, x21 = ret[0]

            fake_ret1 = self.discriminator(x12, 1)
            fake_loss1 = compute_loss(
                fake_ret1, 0, self.criterion)

            fake_ret2 = self.discriminator(x21, 0)
            fake_loss2 = compute_loss(
                fake_ret2, 0, self.criterion)

            fake_loss = fake_loss1 + fake_loss2

            # Total Loss
            result = real_loss + fake_loss

            # Logging
            self.log("loss/disc_real_1", real_loss1)
            self.log("loss/disc_fake_1", fake_loss1)
            self.log("loss/disc_real_2", real_loss2)
            self.log("loss/disc_fake_2", fake_loss2)

        # GENERATOR
        if optimizer_idx == 1:
            ret = self.generator(fake, real)
            x12, x21 = ret[0]
            loss_rec, loss_s, loss_c, loss_cyc = ret[1]

            symmetry_loss = 0
            if self.hparams.use_symmetry_loss:
                r = self.hparams.img_dim[1]
                idx_1 = list(range(0, r//2))
                idx_2 = list(range(r-1, r//2-1, -1))

                a = x12[:, :, idx_1, :]
                b = x12[:, :, idx_2, :]

                symmetry_loss += self.hparams.gamma * \
                    th.mean(th.abs(a - b))

                a = x21[:, :, idx_1, :]
                b = x21[:, :, idx_2, :]

                symmetry_loss += self.hparams.gamma * \
                    th.mean(th.abs(a - b))

            lpips_loss = 0
            if self.hparams.use_lpips:
                pass

                # Gen Loss
            fake_ret2 = self.discriminator(x12, 1)
            fake_loss2 = compute_loss(
                fake_ret2, 1, self.criterion)

            fake_ret1 = self.discriminator(x21, 0)
            fake_loss1 = compute_loss(
                fake_ret1, 1, self.criterion)

            gen_loss = fake_loss1 + fake_loss2

            # Total Loss
            result = self.hparams.l0 * gen_loss + self.hparams.l1 * loss_rec + \
                self.hparams.l2 * loss_s + self.hparams.l3 * \
                loss_c + self.hparams.l4 * loss_cyc + symmetry_loss

            # Logging
            self.log("loss/gen", self.hparams.l0 * gen_loss)
            self.log("loss/rec",  self.hparams.l1 * loss_rec)
            self.log("loss/style", self.hparams.l2 * loss_s)
            self.log("loss/context", self.hparams.l3 * loss_c)
            self.log("loss/cycle",  self.hparams.l4 * loss_cyc)
            if self.hparams.use_symmetry_loss:
                self.log("loss/gen_symmetry", symmetry_loss)

        return result

    def on_epoch_end(self):

        if (self.current_epoch + 1) % self.hparams.fid_interval == 0 or self.current_epoch == 0:

            if self.hparams.moving_average:
                self.generator_avg.eval()
            else:
                self.generator.eval()

            with th.no_grad():
                act_fake = np.zeros_like(self.act_real)

                for i in range(self.n_samples // self.act_samples):
                    # start and end index
                    si = i * self.act_samples
                    ei = min((i+1)*self.act_samples, self.n_samples)

                    # images generated by generator
                    if self.hparams.moving_average:
                        imgs = self.generator_avg(self.gen_input[si:ei])
                    else:
                        imgs = self.generator(self.gen_input[si:ei])

                    # vgg activations
                    act = np.squeeze(vgg16_get_activation_maps(
                        imgs, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
                    act_fake[si:ei] = act

            if self.hparams.moving_average:
                self.generator_avg.train()
            else:
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
