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
        "Generate samples from each epoch for DCGAN and PIX2PIX")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument("mode", choices=["dcgan", "pix2pix"])
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/artificial_cars", help="path to artificial cars")
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
    mode = args.mode
    artificial_dir = args.artificial_dir
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
    if mode == "dcgan":
        interpolate = LatentDimInterpolator(num_samples=10, steps=step)

    # checkpoints
    checkpoints = [
        i-1 for i in range(interval, max_epochs, interval)] + [max_epochs-1]

    for version in os.listdir(os.path.join(experiment_dir, "lightning_logs")):

        print(f"Version : {version}")

        # paths
        checkpoint_path = os.path.join(
            experiment_dir, "lightning_logs", version, "checkpoints", "epoch={}.ckpt")
        results_dir = os.path.join(experiment_dir, version, results_name)
        os.makedirs(results_dir, exist_ok=True)

        # main stuff
        with th.no_grad():
            for epoch in tqdm.tqdm(checkpoints, desc="Checkpoints"):
                path = checkpoint_path.format(epoch)
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

                    # pix2pix needs artificial cars
                    if mode == "pix2pix" and gen_in is None:
                        img_dim = model.hparams.img_dim
                        transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                                        transforms.ToTensor(),
                                                        transforms.RandomHorizontalFlip(
                                                            p=0.5),
                                                        transforms.RandomVerticalFlip(
                                                            p=0.5),
                                                        transforms.Normalize([0.5], [0.5])])

                        artificial_dataset = PostdamCarsDataModule(
                            artificial_dir, transform=transform, img_size=img_dim[1:], batch_size=n_samples)
                        artificial_dataset.setup()
                        dataloader = artificial_dataset.train_dataloader()

                        times = int(
                            np.ceil(n_samples / len(dataloader.dataset)))
                        gen_in = []
                        for _ in range(times):
                            gen_in.append(next(iter(dataloader))[0])
                        gen_in = th.cat(gen_in, dim=0).to(model.device)

                    elif mode == "dcgan" and gen_in is None:
                        gen_in = th.normal(0, truncation, size=(
                            n_samples, model.generator.latent_dim), device=model.device)

                    # fake imgs need to do it in batches for memory stuff
                    fake_imgs = []
                    for i in range((n_samples + bs) // bs):
                        si = i * bs
                        ei = min((i+1)*bs, n_samples)
                        # print(gen_in[si:ei].shape)
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
                        fake_imgs[:n_samples_show], nrow=8, normalize=True)
                    torchvision.utils.save_image(grid, samples_path)

                    # walking through latent space
                    if mode == "dcgan":
                        fake_imgs = interpolate.interpolate_latent_space(
                            model, truncation=truncation)
                        model.eval()

                        grid = torchvision.utils.make_grid(
                            fake_imgs, nrow=step+2, normalize=True)
                        torchvision.utils.save_image(grid, interpolate_path)
                    elif mode == "pix2pix":
                        in_ = gen_in[:10]
                        fake_ = fake_imgs[:10]
                        diff_ = in_ - fake_
                        imgs = th.cat([in_, fake_, diff_], dim=0)

                        grid = torchvision.utils.make_grid(
                            imgs, nrow=10, normalize=True)
                        torchvision.utils.save_image(grid, diff_path)
