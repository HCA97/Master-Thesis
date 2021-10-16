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
    parser.add_argument("checkpoint_path", help="checkpoint path")
    parser.add_argument("save_path", help="save path")
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/artificial_cars", help="path to artificial cars")
    parser.add_argument("--n_samples", default=1000,
                        type=int, help="number of fake images")
    parser.add_argument("--truncation", default=1,
                        type=float, help="dcgan truncation value")
    args = parser.parse_args()

    bs = 500
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    artificial_dir = args.artificial_dir
    n_samples = args.n_samples
    truncation = args.truncation

    os.makedirs(save_path, exist_ok=True)

    model = GAN.load_from_checkpoint(checkpoint_path)
    model.cuda()
    model.eval()

    # pix2pix needs artificial cars
    if model.hparams.gen_model in ["unet", "refiner"]:
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
        gen_in = th.cat(gen_in, dim=0)

    else:
        gen_in = th.normal(0, truncation, size=(
            n_samples, model.generator.latent_dim))

    # fake imgs need to do it in batches for memory stuff
    with th.no_grad():
        fake_imgs = []
        for i in tqdm.tqdm(range(n_samples // bs), desc="Generating"):
            si = i * bs
            ei = min((i+1)*bs, n_samples)
            fake_imgs.append(
                model(gen_in[si:ei].cuda()).detach().cpu().clone())
        fake_imgs = th.cat(fake_imgs, dim=0)

    # save them
    for k, img in tqdm.tqdm(enumerate(fake_imgs), desc="Saving"):
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2
        cv2.imwrite(os.path.join(save_path, f"{k}.png"), cv2.cvtColor(
            255*img, cv2.COLOR_RGB2BGR))
