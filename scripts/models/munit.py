import copy
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
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
                 weight_decay_disc=0,
                 weight_decay_gen=0,
                 # Model Parameters
                 discriminator_params=None,
                 generator_params=None,
                 disc_model="patch",
                 gen_init="gaussian",
                 disc_init="gaussian",
                 # Training Loss
                 use_symmetry_loss=False,
                 gamma=1,
                 use_lpips=False,
                 alpha=1,
                 # Generator Loss
                 l0=1,
                 l1=10,
                 l2=1,
                 l3=1,
                 l4=10,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        generator_params = generator_params if generator_params else {}
        self.generator = MUNITGEN(img_dim, **generator_params)
        self.generator.apply(weights_init(gen_init))

        discriminator_params = discriminator_params if discriminator_params else {}
        self.discriminator = MultiDiscriminator(
            img_dim, disc_model=disc_model, **discriminator_params)
        self.discriminator.apply(weights_init(disc_init))

        self.criterion = L2Norm()

        if use_lpips:
            self.lpips = lpips.LPIPS(net='vgg')

        # ingerating FID score
        self.n_samples = 1024
        self.act_real = []
        self.act_samples = 64
        self.gen_input = []

        # something else
        self.imgs_real = None
        self.imgs_fake = None

    def configure_optimizers(self):

        # optimizers
        opt_disc = th.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate_disc,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay_disc)
        opt_gen = th.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate_gen,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay_gen)

        if self.hparams.use_lr_scheduler:
            dis_sched = StepLR(opt_disc, step_size=100, gamma=0.5)
            gen_sched = StepLR(opt_gen, step_size=100, gamma=0.5)

            return [opt_disc, opt_gen], [dis_sched, gen_sched]

        return [opt_disc, opt_gen]

    def forward(self, tensor1: th.Tensor, tensor2: th.Tensor, inference=True) -> th.Tensor:
        return self.generator(tensor1, tensor2, inference=inference)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, fake = batch

        # append real image activation for FID
        if self.hparams.fid_interval > 0:
            if len(self.act_real) < self.n_samples:
                act = np.squeeze(vgg16_get_activation_maps(
                    real, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
                self.act_real = np.concatenate(
                    (self.act_real, act), axis=0) if len(self.act_real) > 0 else act

            # append fake images or noise vectors for FID
            if len(self.gen_input) < self.n_samples:
                self.gen_input = th.cat((self.gen_input, fake), dim=0) if len(
                    self.gen_input) > 0 else fake

        # store the first batch for callback
        if self.imgs_real is None:
            self.imgs_real = real
        if self.imgs_fake is None:
            self.imgs_fake = fake

        result = None

        # DISCRIMINATOR
        if optimizer_idx == 0:

            # Real Loss
            real_ret1 = self.discriminator(fake, 0)
            real_loss1 = compute_loss(real_ret1, 1, self.criterion)

            real_ret2 = self.discriminator(real, 1)
            real_loss2 = compute_loss(real_ret2, 1, self.criterion)

            real_loss = real_loss1 + real_loss2

            # Fake Loss
            ret = self.generator(fake, real)
            x12, x21 = ret[0]

            fake_ret1 = self.discriminator(x12.detach(), 1)
            fake_loss1 = compute_loss(
                fake_ret1, 0, self.criterion)

            fake_ret2 = self.discriminator(x21.detach(), 0)
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
                lpips_loss += th.mean(self.lpips(fake, x12, normalize=True))
                lpips_loss += th.mean(self.lpips(real, x21, normalize=True))

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
                loss_c + self.hparams.l4 * loss_cyc + \
                self.hparams.gamma * symmetry_loss + self.hparams.alpha * lpips_loss

            # Logging
            self.log("loss/adv_loss", self.hparams.l0 * gen_loss)
            self.log("loss/fake_loss1", self.hparams.l0 * fake_loss1)
            self.log("loss/fake_loss2", self.hparams.l0 * fake_loss2)
            self.log("loss/rec_loss",  self.hparams.l1 * loss_rec)
            self.log("loss/style_loss", self.hparams.l2 * loss_s)
            self.log("loss/content_loss", self.hparams.l3 * loss_c)
            self.log("loss/cycle_loss",  self.hparams.l4 * loss_cyc)
            if self.hparams.use_symmetry_loss:
                self.log("loss/gen_symmetry_loss",
                         self.hparams.gamma * symmetry_loss)
            if self.hparams.use_lpips:
                self.log("loss/gen_lpips_loss",
                         self.hparams.alpha * lpips_loss)

        return result

    def on_epoch_end(self):
        # pass
        if self.hparams.fid_interval > 0 and \
            ((self.current_epoch + 1) % self.hparams.fid_interval == 0 or
             self.current_epoch == 0):

            self.generator.eval()

            if len(self.gen_input) >= self.n_samples:
                with th.no_grad():
                    act_fake = np.zeros_like(self.act_real)

                    for i in range(self.n_samples // self.act_samples):
                        # start and end index
                        si = i * self.act_samples
                        ei = min((i+1)*self.act_samples, self.n_samples)

                        tmp = th.zeros_like(self.gen_input[si:ei])
                        # images generated by generator
                        imgs = self.generator(
                            self.gen_input[si:ei], tmp, inference=True)[0]

                        # vgg activations
                        act = np.squeeze(vgg16_get_activation_maps(
                            imgs, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
                        act_fake[si:ei] = act
                # compute fid
                fid = fid_score(self.act_real, act_fake, skip_vgg=True)

                # log fid
                self.log("fid", fid)

            self.generator.train()
