import os
import pickle
import argparse

import torch as th
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.linear_model import LinearRegression

from scripts import *


@th.no_grad()
def generate_examples(z, U, dims, ratios, n_samples=15, scale=0.2, frequency=0.9):

    z_modified = th.zeros((n_samples+1, len(z)), device=z.device)
    z_modified[0] = z

    for i in range(1, n_samples+1):
        u = th.zeros((U.shape[1],), device=z.device, dtype=z.dtype)
        for dim, (lower, upper) in zip(dims, ratios):
            if np.random.uniform(0, 1) < frequency:
                u[dim] = np.random.uniform(lower*scale, upper*scale)
        z_modified[i] = z + th.matmul(U, u)

    return z_modified


if __name__ == "__main__":
    # input
    gan_path = "experiments/dataaug/lightning_logs/version_1/checkpoints/epoch=774.ckpt"
    encoder_path = "experiments/gan_inversion/dataaug_version_1_epoch=774/iter=300.pkl"
    ganspace_dir = "experiments/ganspace/dataaug_bn_default_epoch=774/linear"

    # load GAN model
    gan = GAN.load_from_checkpoint(gan_path)
    gan.eval()

    # load ENCODER model
    enc = EncoderLatent(
        gan.generator_avg if gan.hparams.moving_average else gan.generator)
    enc.load_state_dict(th.load(encoder_path))
    enc.eval()

    # load RATIOS
    f = open(os.path.join(ganspace_dir, "reg.pkl"), 'rb')
    reg = pickle.load(f)
    f.close()
    U = th.tensor(reg.coef_, device=gan.device, dtype=th.float32)

    data = np.load(os.path.join(ganspace_dir, "important_dims.npy"),
                   allow_pickle=True).item()
    dims = data["dims"]
    ratios = data["ratios"]

    _, w, h = gan.hparams.img_dim

    # PLOTING
    while 1:
        plt.figure(figsize=(10, 5))

        # Fake Example
        z = th.normal(0, 0.75, (gan.generator.latent_dim,), device=gan.device)
        zs = generate_examples(
            z, U, dims, ratios, n_samples=19, scale=0.3, frequency=0.7)
        imgs = gan(zs)

        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
        grid = grid.detach().cpu().numpy().transpose(1, 2, 0)

        plt.subplot(1, 2, 1)
        plt.imshow(grid)
        plt.title("Fake Example", fontsize=18)
        plt.axis('off')
        plt.gca().add_patch(Rectangle((1, 1), h+1, w+1,
                                      linewidth=4, edgecolor='r', facecolor='none'))

        # Real Example
        X_real_path = os.path.join(os.path.split(
            encoder_path)[0], "X_real_val.npy")
        X_real_val = np.load(X_real_path)

        idx = np.random.randint(0, len(X_real_val))
        with th.no_grad():
            img = th.tensor(X_real_val[idx])
            z, _ = enc(th.unsqueeze(img, dim=0))
            zs = generate_examples(
                z[0], U, dims, ratios, n_samples=19, scale=0.3, frequency=0.7)
            imgs = gan(zs)
            imgs[0] = img

            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
            grid = grid.detach().cpu().numpy().transpose(1, 2, 0)

            plt.subplot(1, 2, 2)
            plt.imshow(grid)
            plt.title("Real Example", fontsize=18)
            plt.axis('off')
            plt.gca().add_patch(Rectangle((1, 1), h+1, w+1,
                                          linewidth=4, edgecolor='r', facecolor='none'))

        plt.tight_layout()
        plt.show()

        x = input("Press (q) to quit : ")
        if x == "q":
            break
