import os
import torch as th
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from scripts.callbacks import *
from scripts.models import *
from scripts.dataloader import *


@th.no_grad()
def plot_discriminator_steps(img, pl_module):

    pl_module.eval()

    img_ = np.squeeze(img.detach().cpu().numpy()).transpose((1, 2, 0))
    img_ = (img_ + 1) / 2
    plt.figure()
    plt.imshow(img_)
    plt.axis("off")

    x = img.detach()
    dis_inter_layers = []
    for layer in pl_module.discriminator.conv_blocks:
        x = layer(x)
        if layer.__class__.__name__ == "ConvBlock":
            dis_inter_layers.append(x.detach().clone().permute(1, 0, 2, 3))
    x = x.reshape(1, -1)
    pred = pl_module.discriminator.l1(x).detach().cpu().numpy()

    for i, out in enumerate(dis_inter_layers):
        grid = torchvision.utils.make_grid(
            out, nrow=8, padding=1, normalize=True).numpy().transpose((1, 2, 0))
        plt.figure()
        plt.title(f"Discriminator Layer {i+1}")
        plt.imshow(grid)
        plt.tight_layout()
        plt.axis("off")

    print("Car Prediction is ", pred[0][0])


@th.no_grad()
def plot_generator_steps(z, pl_module):

    pl_module.eval()

    x = pl_module.generator.l1(z).reshape(1, 128, 8, 16).detach()
    gen_inter_layers = []
    for layer in pl_module.generator.conv_blocks:
        x = layer(x)
        if layer.__class__.__name__ in ["ConvBlock", "BatchNorm2d"]:
            gen_inter_layers.append(x.detach().clone().permute(1, 0, 2, 3))

    for i, out in enumerate(gen_inter_layers[:-1]):
        grid = torchvision.utils.make_grid(
            out, nrow=8, padding=1, normalize=True).numpy().transpose((1, 2, 0))
        plt.figure()
        plt.title(f"Generator Layer {i}")
        plt.imshow(grid)
        plt.tight_layout()
        plt.axis("off")

    return gen_inter_layers[-1].permute(1, 0, 2, 3)


checkpoint_path = "experiments/dcgan/lightning_logs/version_0/checkpoints/epoch=499.ckpt"

data_dir = "../potsdam_data/potsdam_cars_test"
potsdam = PostdamCarsDataModule(data_dir, batch_size=1)
potsdam.setup()

model = GAN.load_from_checkpoint(checkpoint_path)
model.eval()

z = th.normal(0, 1, (1, model.hparams.latent_dim),
              device=model.device)

fake_img = plot_generator_steps(z, model)
plot_discriminator_steps(fake_img, model)
real_img, _ = next(iter(potsdam.train_dataloader()))
plot_discriminator_steps(real_img, model)
plt.show()
