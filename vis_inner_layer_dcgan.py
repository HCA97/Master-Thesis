import os
import argparse

import torch as th
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from scripts.callbacks import *
from scripts.models import *
from scripts.dataloader import *


@th.no_grad()
def plot_discriminator_steps(img, pl_module, save_dir):
    pl_module.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Discriminator Inner Layers
    x = img.detach().clone()
    dis_inter_layers = []
    for layer in pl_module.discriminator.conv_blocks:
        x = layer(x)
        if layer.__class__.__name__ == "ConvBlock":
            dis_inter_layers.append(x.detach().clone().permute(1, 0, 2, 3))
    x = x.reshape(1, -1)
    pred = pl_module.discriminator.l1(x).detach().cpu().numpy()

    # Plot Inner Layers
    for i, out in enumerate(dis_inter_layers):
        grid = torchvision.utils.make_grid(
            out, nrow=8, padding=1, normalize=True).numpy().transpose((1, 2, 0))

        plt.figure(figsize=(12, 8))
        plt.title(f"Discriminator Layer {i+1}", fontsize=20)
        plt.imshow(grid)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"disc_layer_{i+1}.png"))

        # close everything, i don't know how important it is
        plt.clf()
        plt.close()

    # Plot Input and Discriminator Prediction
    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1, 2, 0))
    # rescale for matplotlib
    img_ = (img_ + 1) / 2

    plt.figure(figsize=(12, 8))
    plt.imshow(img_)
    plt.axis("off")
    plt.title("Car Prediction is %.3f" % pred.mean(), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "input.png"))

    # close everything, i don't know how important it is
    plt.clf()
    plt.close()


@ th.no_grad()
def plot_generator_steps(z, pl_module, save_dir):
    pl_module.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Generator Inner Layers
    x = pl_module.generator.l1(z).reshape(1, 128, 8, 16).detach()
    gen_inter_layers = []
    for layer in pl_module.generator.conv_blocks:
        x = layer(x)
        if layer.__class__.__name__ in ["ConvBlock", "BatchNorm2d"]:
            gen_inter_layers.append(
                x.detach().cpu().clone().permute(1, 0, 2, 3))

    # Plot Inner Layers
    for i, out in enumerate(gen_inter_layers[:-1]):
        grid = torchvision.utils.make_grid(
            out, nrow=16, padding=1, normalize=True).numpy().transpose((1, 2, 0))

        plt.figure(figsize=(12, 8))
        plt.title(f"Generator Layer {i+1}", fontsize=20)
        plt.imshow(grid)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"gen_layer{i+1}.png"))

        # close everything, i don't know how important it is
        plt.clf()
        plt.close()
    return gen_inter_layers[-1].permute(1, 0, 2, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Show inner layers of DCGAN")
    parser.add_argument(
        "--checkpoint_path", default="experiments/dcgan/lightning_logs/version_0/checkpoints/epoch=499.ckpt", help="path to DCGAN checkpoint")
    parser.add_argument(
        "--data_dir", default="../potsdam_data/potsdam_cars", help="path to real cars directory")
    parser.add_argument("save_dir", help="save directory path")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    data_dir = args.data_dir
    save_dir = args.save_dir

    # data loader
    potsdam = PostdamCarsDataModule(data_dir, batch_size=1)
    potsdam.setup()

    # load model
    model = GAN.load_from_checkpoint(checkpoint_path)
    model.eval()

    # noise
    z = th.normal(0, 1, (1, model.hparams.latent_dim),
                  device=model.device)

    # fake imgs
    fake_img = plot_generator_steps(
        z, model, os.path.join(save_dir, "fake_img"))
    plot_discriminator_steps(
        fake_img, model, os.path.join(save_dir, "fake_img"))

    # real imgs
    real_img, _ = next(iter(potsdam.train_dataloader()))
    plot_discriminator_steps(
        real_img, model, os.path.join(save_dir, "real_img"))
    # plt.show()
