import os
import argparse

import torch as th
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from scripts.callbacks import *
from scripts.models import *
from scripts.dataloader import *

NUM_ROWS = {1024: 64, 512: 32, 256: 16, 128: 16, 64: 8, 32: 8, 16: 4}
FIG_DIM = {1024: (12, 6), 512: (12, 6), 256: (
    12, 8), 128: (12, 6), 64: (12, 8), 32: (12, 6), 16: (12, 6)}


def plot_layer_disc(out, layer_i, save_dir):
    out = out.detach().cpu().clone().permute(1, 0, 2, 3)
    grid = torchvision.utils.make_grid(
        out, nrow=NUM_ROWS[out.shape[0]], padding=1, normalize=True, pad_value=1).numpy().transpose((1, 2, 0))

    plt.figure(figsize=FIG_DIM[out.shape[0]])
    # plt.title(f"Discriminator Layer {layer_i+1}", fontsize=20)
    plt.imshow(grid)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"disc_layer_{layer_i+1}.png"))

    # close everything, i don't know how important it is
    plt.clf()
    plt.close()


def plot_layer(out, layer_i, save_dir, which_one):
    out = out.detach().cpu().clone().permute(1, 0, 2, 3)
    grid = torchvision.utils.make_grid(
        out, nrow=NUM_ROWS[out.shape[0]], padding=1, normalize=True, pad_value=1).numpy().transpose((1, 2, 0))

    plt.figure(figsize=FIG_DIM[out.shape[0]])
    plt.imshow(grid)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{which_one}_layer{layer_i+1}.png"))

    # # close everything, i don't know how important it is
    plt.clf()
    plt.close()

#
#
#


@th.no_grad()
def plot_discriminator_steps(img, pl_module, save_dir):
    pl_module.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Discriminator Inner Layers
    x = img.detach().clone()
    dis_inter_layers = []
    for i, layer in enumerate(pl_module.discriminator.conv_blocks):
        x = layer(x)
        if layer.__class__.__name__ in ["ConvBlock", "AvgPool2d"]:
            plot_layer(x, i, save_dir, "disc")

    x = x.reshape(1, -1)
    pred = pl_module.discriminator.l1(x).detach().cpu().numpy()

    # Plot Input and Discriminator Prediction
    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1, 2, 0))
    # rescale for matplotlib
    img_ = (img_ - min(-1, img_.min())) / \
        (max(1, img_.max()) - min(-1, img_.min()))

    plt.figure(figsize=(12, 8))
    plt.imshow(img_)
    plt.axis("off")
    plt.title("Car Prediction is %.3f" % pred.mean(), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "input.png"))

    # close everything, i don't know how important it is
    plt.clf()
    plt.close()


@th.no_grad()
def plot_discriminator_steps_patch(img, pl_module, save_dir):
    pl_module.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Discriminator Inner Layers
    x = img.detach().clone()
    for i, layer in enumerate(pl_module.discriminator.conv_blocks):
        x = layer(x)
        if layer.__class__.__name__ in ["ConvBlock", "ResBlock"]:
            plot_layer(x, i, save_dir, "disc")

    pred = pl_module.discriminator.l1(x).detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.title(f"Discriminator Final", fontsize=20)
    colorbar = plt.imshow(np.squeeze(pred), cmap="gray")
    plt.colorbar(colorbar)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"disc_final_layer.png"))

    # close everything, i don't know how important it is
    plt.clf()
    plt.close()

    # Plot Input and Discriminator Prediction
    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1, 2, 0))
    # rescale for matplotlib
    img_ = (img_ - min(-1, img_.min())) / \
        (max(1, img_.max()) - min(-1, img_.min()))

    plt.figure(figsize=(12, 8))
    plt.imshow(img_)
    plt.axis("off")
    plt.title("Car Prediction is %.3f" % pred.mean(), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "input.png"))

    # close everything, i don't know how important it is
    plt.clf()
    plt.close()


@th.no_grad()
def plot_discriminator_steps_basic_patch(img, pl_module, save_dir):

    class A:
        def __init__(self, disc):
            self.discriminator = disc

        def eval(self):
            self.discriminator.eval()

        def train(self):
            self.discriminator.train()

    plot_discriminator_steps_patch(
        img, A(pl_module.discriminator.patch), os.path.join(save_dir, "patch"))
    plot_discriminator_steps(
        img,  A(pl_module.discriminator.basic), os.path.join(save_dir, "basic"))


@ th.no_grad()
def plot_generator_steps(z, pl_module, save_dir):
    pl_module.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Generator Inner Layers
    init_channels = pl_module.generator.init_channels
    init_height = pl_module.generator.init_height
    init_width = pl_module.generator.init_width
    x = pl_module.generator.l1(z).reshape(
        1, init_channels, init_height, init_width).detach()
    gen_inter_layers = []
    layer_i = 0
    for layer in pl_module.generator.conv_blocks[:-1]:
        x = layer(x)
        if layer.__class__.__name__ in ["ConvBlock", "BatchNorm2d", "ResBlock"]:
            plot_layer(x, layer_i, save_dir, "gen")
            layer_i += 1
    x = pl_module.generator.conv_blocks[-1](x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Show inner layers of DCGAN")
    parser.add_argument("save_dir", help="save directory path")
    parser.add_argument("checkpoint_dir", help="path to DCGAN checkpoint")
    parser.add_argument(
        "--data_dir", default="../potsdam_data/potsdam_cars_corrected", help="path to real cars directory")
    parser.add_argument("--artificial_dir",
                        default="../potsdam_data/artificial_cars", help="path to artificial cars")
    parser.add_argument("--n_samples", default=10,
                        type=int, help="number of samples")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_dir
    data_dir = args.data_dir
    artificial_dir = args.artificial_dir
    save_dir = args.save_dir
    n_samples = args.n_samples

    # load model
    model = GAN.load_from_checkpoint(checkpoint_path)
    model.eval()

    # data loader
    potsdam = PostdamCarsDataModule(
        data_dir, data_dir2=artificial_dir, img_size=model.hparams.img_dim[1:], batch_size=1)
    potsdam.setup()

    for i in range(1, n_samples+1):
        real_img, fake_img_ = next(iter(potsdam.train_dataloader()))

        if model.hparams.gen_model in ["basic", "resnet"]:
            z = th.normal(0, 1, (1, model.generator.latent_dim),
                          device=model.device)
            fake_img = plot_generator_steps(
                z, model, os.path.join(save_dir, str(i), "fake_img"))

        if model.hparams.disc_model in ["basic", "resnet"]:
            plot_discriminator_steps(
                fake_img, model, os.path.join(save_dir, str(i), "fake_img"))
            plot_discriminator_steps(
                real_img, model, os.path.join(save_dir, str(i), "real_img"))
        elif model.hparams.disc_model == "patch":
            plot_discriminator_steps_patch(
                fake_img, model, os.path.join(save_dir, str(i), "fake_img"))
            plot_discriminator_steps_patch(
                real_img, model, os.path.join(save_dir, str(i), "real_img"))
        elif model.hparams.disc_model == "basicpatch":
            plot_discriminator_steps_basic_patch(
                fake_img, model, os.path.join(save_dir, str(i), "fake_img"))
            plot_discriminator_steps_basic_patch(
                real_img, model, os.path.join(save_dir, str(i), "real_img"))
