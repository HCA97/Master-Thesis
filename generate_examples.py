import os
import argparse

import tqdm
import torch as th
import torchvision
from torchvision import transforms

import cv2
import numpy as np

from scripts import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Generate samples from each epoch for DCGAN")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--n_samples", default=1024,
                        type=int, help="number of fake images")
    parser.add_argument("--truncation", default=1,
                        type=float, help="dcgan truncation value")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    n_samples = args.n_samples
    interval = args.interval
    truncation = args.truncation
    results_name = args.results_name

    # other
    step = 5
    n_samples_show = 64
    gen_in = None
    max_epochs = 1000
    bs = 256

    # for interpolation between points
    interpolate = LatentDimInterpolator(num_samples=10, steps=step)

    for version in os.listdir(os.path.join(experiment_dir, "lightning_logs")):

        print(f"Version : {version}")

        # paths
        checkpoint_path = os.path.join(
            experiment_dir, "lightning_logs", version, "checkpoints")
        checkpoints = get_epoch_number(checkpoint_path)

        results_dir = os.path.join(experiment_dir, version, results_name)
        os.makedirs(results_dir, exist_ok=True)

        # main stuff
        with th.no_grad():
            for epoch in tqdm.tqdm(checkpoints, desc="Checkpoints"):
                path = os.path.join(checkpoint_path, f"epoch={epoch}.ckpt")
                if os.path.exists(path):
                    # load model
                    model = GAN.load_from_checkpoint(path)
                    model.eval()
                    model.cuda()

                    # paths
                    img_folder = os.path.join(
                        results_dir, "images", f"epoch={epoch}")
                    interpolate_path = os.path.join(
                        results_dir, f"interpolation_epoch={epoch}.png")
                    samples_path = os.path.join(
                        results_dir, f"epoch={epoch}.png")
                    diff_path = os.path.join(
                        results_dir, f"diff_epoch={epoch}.png")

                    # skip to speed up the computation
                    if os.path.exists(interpolate_path) or os.path.exists(diff_path):
                        continue

                    os.makedirs(img_folder, exist_ok=True)

                    gen_in = th.normal(0, truncation, size=(
                        n_samples, model.generator.latent_dim), device=model.device)

                    # fake imgs need to do it in batches for memory stuff
                    fake_imgs = []
                    for i in range((n_samples + bs) // bs):
                        si = i * bs
                        ei = min((i+1)*bs, n_samples)
                        fake_imgs.append(
                            model(gen_in[si:ei]).detach().cpu().clone())
                    fake_imgs = th.cat(fake_imgs, dim=0)

                    # save them
                    for k, img in enumerate(fake_imgs):
                        img = img.detach().cpu().numpy().transpose(1, 2, 0)
                        img = (img + 1) / 2
                        cv2.imwrite(os.path.join(img_folder, f"{k}.png"), cv2.cvtColor(
                            255*img, cv2.COLOR_RGB2BGR))

                    # show small sample for visualization
                    grid = torchvision.utils.make_grid(
                        fake_imgs[:n_samples_show], nrow=8, normalize=True, pad_value=1)
                    torchvision.utils.save_image(grid, samples_path)

                    # walking through latent space
                    fake_imgs = interpolate.interpolate_latent_space(
                        model, truncation=truncation)
                    model.eval()

                    grid = torchvision.utils.make_grid(
                        fake_imgs, nrow=step+2, normalize=True, pad_value=1)
                    torchvision.utils.save_image(grid, interpolate_path)
