import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                 betas=(0.5, 0.999),
                 discriminator_params=None,
                 generator_params=None,
                 use_gp=False,
                 alpha=1.0,
                 beta=1.0,
                 gen_model="basic",
                 disc_model="basic",
                 fid_interval=5,
                 use_lr_scheduler=False,
                 gen_init="normal",
                 disc_init="normal"):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self.get_generator(generator_params)
        self.discriminator = self.get_discriminator(discriminator_params)
        self.criterion = self.get_criterion()

        # TODO WEIGHT AVERAGIN FOR GEN
        # TODO BUFFER FOR DISC

        # ingerating FID score
        self.n_samples = 1024
        self.act_real = []
        self.act_samples = 512
        self.gen_input = []
        self.imgs_real = None

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
        elif self.hparams.gen_init == "default":
            pass
        return generator

    def get_discriminator(self, discriminator_params):
        discriminator_params = discriminator_params if discriminator_params else {}

        if self.hparams.disc_model == "basic":
            discriminator = BasicDiscriminator(
                self.hparams.img_dim, **discriminator_params)
        else:
            raise NotImplementedError()

        if self.hparams.disc_init == "normal":
            discriminator.apply(weights_init_normal)
        return discriminator

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = self.hparams.betas

        # optimizers
        opt_disc = th.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = th.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)

        # scheduler - patience is 125 epochs
        # pytorch lighting is annoying
        patience = 125 // self.hparams.fid_interval
        self.scheduler_disc = ReduceLROnPlateau(
            opt_disc, 'min', patience=patience, factor=0.5, verbose=True)
        self.scheduler_gen = ReduceLROnPlateau(
            opt_gen, 'min', patience=patience, factor=0.5, verbose=True)
        return [opt_disc, opt_gen], []

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        return self.generator(tensor)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, fake = batch

        # append real image act and fake image
        if len(self.act_real) < self.n_samples:
            act = np.squeeze(vgg16_get_activation_maps(
                real, layer_idx=33, device="cpu", normalize_range=(-1, 1)).numpy())
            self.act_real = np.concatenate(
                (self.act_real, act), axis=0) if len(self.act_real) > 0 else act

        # append fake images or noise vectors
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
            if self.hparams.gen_model in ["unet", "refiner"]:
                fake_ = self.generator(fake)
            else:
                z = th.normal(0, 1, size=(
                    len(real), self.generator.latent_dim), device=self.device)
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
            if self.hparams.gen_model in ["unet", "refiner"]:
                fake_ = self.generator(fake)

                # L1-Norm
                l1_loss = self.hparams.beta * th.sum(th.abs(fake - fake_))
            else:
                z = th.normal(0, 1, size=(
                    len(real), self.generator.latent_dim), device=self.device)
                fake_ = self.generator(z)

            # Gen Loss
            fake_pred = self.discriminator(fake_)
            fake_gt = th.ones_like(fake_pred)
            gen_loss = self.criterion(fake_pred, fake_gt)

            # Total Loss
            result = gen_loss + l1_loss

            # Logging
            self.log("loss/gen", gen_loss)
            if self.hparams.gen_model in ["unet", "refiner"]:
                self.log("loss/gen_l1", l1_loss)

        return result

    def on_epoch_end(self):

        if (self.current_epoch + 1) % self.hparams.fid_interval == 0 or self.current_epoch == 0:
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
            opt = self.optimizers(use_pl_optimizer=True)[0]
            self.log("lr", opt.param_groups[0]["lr"])
