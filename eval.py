
import os
import torch as th
import torchvision
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from scripts.callbacks import *
from scripts.models import *

checkpoints = [i for i in range(0, 500, 50)] + [499]
checkpoint_path = "experiments/dcgan/lightning_logs/version_0/checkpoints/epoch={}.ckpt"
results_dir = "experiments/dcgan/results"
num_samples = 64
steps = 5

os.makedirs(results_dir, exist_ok=True)

fake_imgs = []
inter_imgs = []

# generate images from same latent points
interpolate = LatentDimInterpolator(num_samples=10, steps=steps)
z = None

for i in checkpoints:
    model = GAN.load_from_checkpoint(checkpoint_path.format(i)).cuda()
    model.eval()

    img_name = os.path.join(results_dir, f"images_epoch={i}.png")
    inter_name = os.path.join(results_dir, f"interpolation_epoch={i}.png")

    # generate examples
    if z is None:
        z = th.normal(0, 1, (num_samples, model.hparams.latent_dim, 1, 1),
                      device=model.device)
    images = model(z)
    grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    grid = (grid.cpu().detach().numpy().transpose(
        (1, 2, 0))*255).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid)
    plt.title(f"Epoch {i}", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.clf()
    plt.close()

    fake_imgs.append(imageio.imread(img_name))

    # interpolate
    # interpolate = LatentDimInterpolator(num_samples=10, steps=steps)
    images = interpolate.interpolate_latent_space(
        model, model.hparams.latent_dim)
    model.eval()

    grid = torchvision.utils.make_grid(
        images, nrow=steps+2, normalize=True)
    grid = (grid.cpu().detach().numpy().transpose(
        (1, 2, 0))*255).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid)
    plt.title(f"Epoch {i}", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(inter_name)
    plt.clf()
    plt.close()

    inter_imgs.append(imageio.imread(inter_name))

imageio.mimsave(os.path.join(results_dir, "images.gif"),
                fake_imgs, fps=1)

imageio.mimsave(os.path.join(results_dir, "interpolation.gif"),
                inter_imgs, fps=1)

print(summary(model.discriminator, (3, 32, 63)))
print(summary(model.generator, (100, 1, 1)))
