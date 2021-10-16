import os
import argparse
import csv

import torch as th
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from scripts import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Computes FID score multiple times. Make sure you run `eval_gan.py` first")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument(
        "--potsdam_train_dir", default="../potsdam_data/potsdam_cars_all", help="path to potsdam training cars")
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/artificial_cars", help="path to artificial cars")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    parser.add_argument("--data_aug", type=int, default=0,
                        help="data augmentation types")
    parser.add_argument("--width", type=int, default=64, help="image width")
    parser.add_argument("--height", type=int, default=32, help="image height")
    args = parser.parse_args()

    # other args
    fid_samples = 1024
    img_dim = (args.height, args.width)
    device = "cuda:0"

    # data aug
    brightness = [1, 1.1]
    contrast = [1, 1.25]
    hue = [-0.1, 0.1]
    data_augs = [
        # no dat aug
        transforms.Compose([transforms.Resize(img_dim),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]),
        # horizontal flip only
        transforms.Compose([transforms.Resize(img_dim),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.Normalize([0.5], [0.5])]),
        # horizontal flip + hue
        transforms.Compose([transforms.Resize(img_dim),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ColorJitter(hue=hue),
                            transforms.Normalize([0.5], [0.5])]),
        # horizontal flip + hue + contrast
        transforms.Compose([transforms.Resize(img_dim),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ColorJitter(hue=hue, contrast=contrast),
                            transforms.Normalize([0.5], [0.5])]),
        # horizontal flip + hue + contrast + brightness
        transforms.Compose([transforms.Resize(img_dim),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ColorJitter(
                                hue=hue, brightness=brightness, contrast=contrast),
                            transforms.Normalize([0.5], [0.5])]),
    ]

    # data loaders
    dataset = ImageFolder(args.potsdam_train_dir, root2=args.artificial_dir,
                          transform2=data_augs[1], transform=data_augs[args.data_aug])
    potsdam_cars_dataloader = DataLoader(
        dataset, batch_size=fid_samples, shuffle=True, num_workers=4)

    csv_rows = []

    for version in os.listdir(args.experiment_dir):

        # skip logs
        if version == "lightning_logs":
            continue

        # skip everything that is a file
        if os.path.isfile(os.path.join(args.experiment_dir, version)):
            continue

        # load fid scores if there
        numpy_file = os.path.join(
            args.experiment_dir, version, args.results_name, "fid_scores.npy")
        if os.path.exists(numpy_file):
            data = np.load(numpy_file, allow_pickle=True).item()

            fid_train = data.get("fid_train", [])
            epochs = data.get("epochs", [])

            if fid_train and epochs:
                # find the best epoch with lowest fid
                idx = np.argmin(fid_train)
                checkpoint = epochs[idx]

                path = os.path.join(
                    args.experiment_dir, "lightning_logs", version, "checkpoints", f"epoch={checkpoint}.ckpt")

                model = GAN.load_from_checkpoint(path)
                model.to(device)
                model.eval()

                row = {"version": version, "potsdam_dir": args.potsdam_train_dir,
                       "artificial_dir": args.artificial_dir, "dataaug": args.data_aug, "best_epoch": checkpoint}
                with th.no_grad():
                    fids = []
                    for i in range(5):
                        real, artificial = next(iter(potsdam_cars_dataloader))

                        if model.hparams.gen_model in ["unet", "refiner"]:
                            fake = model(artificial.to(model.device))
                        else:
                            z = th.normal(
                                0, 1, (fid_samples, model.generator.latent_dim), device=model.device)
                            fake = model(z)

                        fids.append(
                            fid_score(real, fake, device=device, memorized_fid=False))

                        row["FID_%d" % i] = fids[-1]
                    row["mean_FID"] = np.mean(fids)
                    row["std_FID"] = np.std(fids)

                csv_rows.append(row)

        # save csv
        if csv_rows:
            with open(os.path.join(args.experiment_dir, "fid_results.csv"), "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
