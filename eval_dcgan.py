import os
import argparse

import torch as th
import torchvision
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from scripts import *

parser = argparse.ArgumentParser("Evaluation script for DCGAN")
parser.add_argument("--checkpoint_dir",
                    default="experiments/dcgan/lightning_logs/version_0/checkpoints", help="path to checkpoints")
parser.add_argument(
    "--potsdam_dir", default="../potsdam_data/potsdam_cars", help="path to potsdam cars")
parser.add_argument("results_dir", help="where to save the results")
parser.add_argument("--interval", default=50, type=int,
                    help="interval between two epochs")
parser.add_argument("--use_bn", action="store_true",
                    help="use vgg with bn or not", dest="use_bn")
parser.set_defaults(use_bn=False)
args = parser.parse_args()

layer_idx = 33 if args.use_bn else 24
interval = args.interval
checkpoint_dir = args.checkpoint_dir
potsdam_dir = args.potsdam_dir
results_dir = args.results_dir

checkpoints = [i-1 for i in range(interval, 500, interval)] + [499]
checkpoint_path = os.path.join(checkpoint_dir, "epoch={}.ckpt")

# other arguments
num_samples = 64
steps = 5

os.makedirs(results_dir, exist_ok=True)

fake_imgs = []
inter_imgs = []

# generate images from same latent points
interpolate = LatentDimInterpolator(num_samples=10, steps=steps)
z = None

# fid score stuff
potsdam_dataset = PostdamCarsDataModule(potsdam_dir, batch_size=1024)
potsdam_dataset.setup()
dataloader = potsdam_dataset.train_dataloader()
imgs2 = next(iter(dataloader))[0]
imgs1 = next(iter(dataloader))[0]
# # 33 => bn, 24 => normal
fid = fid_score(imgs1, imgs2, device="cuda:0",
                layer_idx=layer_idx, use_bn=args.use_bn)
print(f"FID score of training set is {fid}.")
# raise RuntimeError
del imgs1
z_fid = None
fid_scores = []

with th.no_grad():
    for i in checkpoints:
        model = GAN.load_from_checkpoint(checkpoint_path.format(i))
        model.eval()

        img_name = os.path.join(results_dir, f"images_epoch={i}.png")
        inter_name = os.path.join(results_dir, f"interpolation_epoch={i}.png")

        # compute FID score
        if z_fid is None:
            z_fid = th.normal(
                0, 1, (1024, model.hparams.latent_dim, 1, 1), device=model.device)
        imgs1 = model(z_fid)
        fid = fid_score(imgs1, imgs2, device="cuda:0",
                        layer_idx=layer_idx, use_bn=args.use_bn)
        del imgs1
        print(f"Epoch {i} - FID {fid}")
        # raise RuntimeError()
        fid_scores.append(fid)

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

plt.figure()
plt.plot(checkpoints, fid_scores, marker="*")
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("FID Score", fontsize=18)
plt.title("FID Score", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "fid_score.png"))

# save them as gif
imageio.mimsave(os.path.join(results_dir, "images.gif"),
                fake_imgs, fps=1)

imageio.mimsave(os.path.join(results_dir, "interpolation.gif"),
                inter_imgs, fps=1)

# print(summary(model.discriminator.cuda(), (3, 32, 62)))
# print(summary(model.generator.cuda(), (100, 1, 1)))
