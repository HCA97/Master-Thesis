import os
import argparse

import tqdm
import torch as th
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scripts import *

# save plots for latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Evaluation script for DCGAN and MUNIT")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument(
        "--potsdam_train_dir", default="../potsdam_data/potsdam_cars_corrected", help="path to potsdam training cars")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--skip_train_fid", action="store_true",
                        help="skip fid of training dataset", dest="skip_train_fid")
    parser.add_argument("--data_aug", type=int, default=0,
                        help="data augmentation types")
    parser.add_argument("--width", type=int, default=64, help="image width")
    parser.add_argument("--height", type=int, default=32, help="image height")
    parser.set_defaults(skip_train_fid=False)
    parser.set_defaults(skip_val=False)
    args = parser.parse_args()

    # other args
    max_epochs = 1000
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
        # MUNIT
        transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                            DynamicPad(min_img_dim=(110, 60),
                                       padding_mode="edge"),
                            transforms.RandomCrop(
                                (55, 105), padding_mode="reflect"),
                            transforms.Resize(img_dim),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]),
        # Heavy Data Aug
        transforms.Compose([transforms.ColorJitter(hue=[-0.2, 0.2], contrast=(1, 1.2)),
                            DynamicPad(min_img_dim=(128, 64),
                                       padding_mode="edge"),
                            transforms.RandomAffine(degrees=10,
                                                    translate=(
                                                        0.075, 0.075),
                                                    scale=(1, 1.2),
                                                    shear=(
                                                        -3, 3, -3, 3),
                                                    interpolation=PIL.Image.BICUBIC,
                                                    fill=100),
                            transforms.RandomAdjustSharpness(2, p=0.5),
                            transforms.Resize((64, 128)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])])
    ]

    # # checkpoints
    # checkpoints = get_epoch_number(os.path.join(
    #     args.experiment_dir, version, args.results_name, "images"))

    # data loaders
    dataset = ImageFolder(args.potsdam_train_dir,
                          transform=data_augs[args.data_aug])

    potsdam_cars_dataloader = DataLoader(
        dataset, batch_size=fid_samples, shuffle=True, num_workers=4)

    img_train1, _ = next(iter(potsdam_cars_dataloader))

    if not args.skip_train_fid:
        img_train2, _ = next(iter(potsdam_cars_dataloader))
        fid1 = fid_score(img_train1, img_train2, device=device)
        print(f"[DATASET] Train FID is {fid1}")

    # save act so we don't need to compute every iteration
    act_train = vgg16_get_activation_maps(img_train1, 33, device, (-1, 1))

    for version in tqdm.tqdm(os.listdir(args.experiment_dir), desc="Version"):

        if version == "lightning_logs":
            continue

        if os.path.isfile(os.path.join(args.experiment_dir, version)):
            continue

        checkpoints = get_epoch_number(os.path.join(
            args.experiment_dir, version, args.results_name, "images"))

        print(f"[FAKE] Loading {version}")
        # evaluation
        fid_train = []

        numpy_file = os.path.join(
            args.experiment_dir, version, args.results_name, "fid_scores.npy")
        if os.path.exists(numpy_file):
            data = np.load(numpy_file, allow_pickle=True).item()

            # load data
            fid_train = data.get("fid_train", [])

        if not fid_train:
            for checkpoint in checkpoints:
                # fake path
                path = os.path.join(
                    args.experiment_dir, version, args.results_name, "images", f"epoch={checkpoint}")

                if not os.path.isdir(path):
                    continue

                dataset = ImageFolder(path, transform=data_augs[0])
                fake_cars_dataloader = DataLoader(
                    dataset, batch_size=fid_samples, num_workers=4)

                img_fake = next(iter(fake_cars_dataloader))[0]
                act_fake = vgg16_get_activation_maps(
                    img_fake, 33, device, (-1, 1))

                # fid score
                fid_train.append(fid_score(act_train, act_fake,
                                 skip_vgg=True, n_cases=fid_samples))
                print(
                    f"[FAKE] Epoch {checkpoint}: Train FID is {fid_train[-1]}.")

            np.save(numpy_file, {"epochs": checkpoints,
                    "fid_train": fid_train})

        # PLOT STUFF
        plt.figure()
        plt.plot(checkpoints[:len(fid_train)], fid_train, marker="*")
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("FID Score", fontsize=18)
        # plt.title("Training FID Score", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.experiment_dir, version,
                    args.results_name, "fid_score_train.pgf"))
        plt.savefig(os.path.join(args.experiment_dir, version,
                    args.results_name, "fid_score_train.png"))
        # close everything, i don't know how important it is
        plt.clf()
        plt.close()
