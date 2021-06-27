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
                 gen_model="basic",
                 disc_model="basic",
                 gen_init="normal",
                 disc_init="normal",
                 kappa=1,
                 # GAN Loss
                 use_symmetry_loss=False,
                 gamma=1,
                 # For pix2pix reconstruction loss
                 rec_loss="l1",
                 beta=1.0,
                 # Training Strategies
                 use_gp=False,
                 alpha=1.0,
                 one_sided_label_smoothing=False,
                 moving_average=False,
                 use_buffer=False,
                 buffer_method="random",
                 buffer_size=1000):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = self.get_criterion()

        if moving_average:
            self.generator_avg = self.get_generator(generator_params)
            self.generator_avg.load_state_dict(
                copy.deepcopy(self.generator.state_dict()))

        # if rec_loss == "lpips":
        #     self.lpips = lpips.LPIPS(net='vgg')

        if use_buffer:
            if buffer_size % 100 != 0:
                raise AttributeError(
                    "Length of the buffer must be divisible by 100.")
            self.buffer = [None for i in range(buffer_size)]

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

        # moving average
        if self.hparams.moving_average and optimizer_idx == 1:
            for p, avg_p in zip(self.generator.parameters(), self.generator_avg.parameters()):
                avg_p.data.copy_(0.999*avg_p.data + 0.001*p.data)

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

        # Train discriminator
        if optimizer_idx == 0:
            if self.hparams.use_gp:
                real = real.requires_grad_(True)

            # Train with real
            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                real_pred, real_patch = self.discriminator(real)
            else:
                real_pred = self.discriminator(real)

            # Loss
            real_gt = th.ones_like(real_pred)
            # Noisy Labels
            if self.hparams.one_sided_label_smoothing:
                real_gt = 0.9*real_gt
            real_loss = self.criterion(real_pred, real_gt)
            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                real_patch_gt = th.ones_like(real_patch)
                real_loss += self.hparams.kappa * \
                    self.criterion(real_patch, real_patch_gt)

            # Gradient Penalty (R1)
            gp_loss = 0
            if self.hparams.use_gp:
                grad = th.autograd.grad(real_pred, real, grad_outputs=th.ones(real_pred.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                slopes = th.sqrt(th.sum(th.mul(grad, grad), dim=[1, 2, 3]))
                gp_loss = 0.5 * self.hparams.alpha * th.mean(slopes)

            # Train with fake
            if self.hparams.gen_model in ["unet", "refiner"]:
                fake_ = self.generator(fake)
            else:
                z = th.normal(0, 1, size=(
                    len(real), self.generator.latent_dim), device=self.device)
                fake_ = self.generator(z)

            # Buffer
            if self.hparams.use_buffer:
                # copy the batch so we can copy it to buffer later
                fake_copy = fake_.detach().cpu().clone()

                # skip buffer until it is full
                try:
                    self.buffer.index(None)
                except ValueError:
                    pick = len(fake_) // 2
                    idx = np.random.choice(
                        len(self.buffer), pick, replace=False)

                    for i in range(pick):
                        fake_[i] = self.buffer[idx[i]].to(self.device)

            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                fake_pred, fake_patch = self.discriminator(fake_)
            else:
                fake_pred = self.discriminator(fake_)

            if self.hparams.use_buffer:
                # remember up to 100 iterations
                cases = len(self.buffer) // 100

                if len(fake_copy) < cases:
                    raise RuntimeError(
                        f"If you are using ``use_buffer`` the batch size must be larger than {cases}. Either decrease the size of your buffer or increase your batch size.")

                # slice in the buffer, not the best but it works
                si = cases * (self.global_step % 100)
                ei = min(si + cases, len(self.buffer))  # incase it overshoots

                # copy predictions, i don't know this much copying is needed
                fake_pred_copy = fake_pred.detach().cpu().clone()
                # fake_pred_copy = fake_pred_copy.mean(axis=1)
                fake_pred_copy = fake_pred_copy.mean(axis=1) if len(
                    fake_pred_copy.shape) == 2 else fake_pred_copy.mean(axis=(1, 2, 3))
                idx = th.argsort(fake_pred_copy)  # ascending order

                # store the buffer
                pick = ei - si  # incase it overshoots
                if self.hparams.buffer_method == "best":
                    self.buffer[si:ei] = fake_copy[idx[-pick:]]
                elif self.hparams.buffer_method == "worst":
                    self.buffer[si:ei] = fake_copy[idx[:pick]]
                else:
                    self.buffer[si:ei] = fake_copy[:pick]

            # Loss
            fake_gt = th.zeros_like(fake_pred)
            fake_loss = self.criterion(fake_pred, fake_gt)
            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                fake_patch_gt = th.zeros_like(fake_patch)
                fake_loss += self.hparams.kappa * \
                    self.criterion(fake_patch, fake_patch_gt)

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
            if self.hparams.gen_model in ["unet", "refiner"]:
                fake_ = self.generator(fake)

                # L1-Norm
                if self.hparams.rec_loss == "l1":
                    l1_loss = self.hparams.beta * th.sum(th.abs(fake - fake_))
                elif self.hparams.rec_loss == "l1_channel_avg":
                    l1_loss = self.hparams.beta * \
                        th.sum(th.abs(fake.mean(axis=(1)) -
                               fake_.mean(axis=(1))))
                # else:
                #     # this might use a lot of vram
                #     self.lpips.to(self.device)
                #     l1_loss = self.hparams.beta * \
                #         th.sum(self.lpips(fake, fake_, normalize=True))
            else:
                z = th.normal(0, 1, size=(
                    len(real), self.generator.latent_dim), device=self.device)
                fake_ = self.generator(z)

            symmetry_loss = 0
            if self.hparams.use_symmetry_loss:
                r = self.hparams.img_dim[1]
                idx_1 = list(range(0, r//2))
                idx_2 = list(range(r-1, r//2-1, -1))

                fake_1 = fake_[:, :, idx_1, :]
                fake_2 = fake_[:, :, idx_2, :]

                symmetry_loss = self.hparams.gamma * \
                    th.mean(th.abs(fake_1 - fake_2))

            # Gen Loss
            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                fake_pred, fake_patch = self.discriminator(fake_)
            else:
                fake_pred = self.discriminator(fake_)
            fake_gt = th.ones_like(fake_pred)

            gen_loss = self.criterion(fake_pred, fake_gt)
            if self.hparams.disc_model == "basic" and self.discriminator.heat_map:
                fake_patch_gt = th.ones_like(fake_patch)
                gen_loss += self.hparams.kappa * \
                    self.criterion(fake_patch, fake_patch_gt)

            # Total Loss
            result = gen_loss + l1_loss + symmetry_loss

            # Logging
            self.log("loss/gen", gen_loss)
            if self.hparams.gen_model in ["unet", "refiner"]:
                self.log("loss/gen_l1", l1_loss)
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
