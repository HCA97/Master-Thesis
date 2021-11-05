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
        "Generate samples from each checkpoint for MUNIT")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument(
        "--potsdam_dir", default="../potsdam_data/potsdam_cars_corrected", help="path to potsdam cars")
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/artificial_cars", help="path to artificial cars")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    parser.add_argument("--use_edges", action="store_true")
    parser.add_argument("--use_dynamic_padding", action="store_true")
    parser.set_defaults(use_dynamic_padding=False)
    parser.set_defaults(use_edges=False)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    potsdam_dir = args.potsdam_dir
    artificial_dir = args.artificial_dir
    interval = args.interval
    results_name = args.results_name

    # other
    img_real = None
    img_fake = None
    random_styles_fake = None
    random_styles_real = None
    n_samples = 10

    max_epochs = 1000

    artificial_transform = [transforms.Resize((32, 64)),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.Normalize([0.5], [0.5])]
    if args.use_edges:
        artificial_transform.insert(1, Skeleton(1, 20, True))

    if args.use_dynamic_padding:
        artificial_transform.insert(0, DynamicPad(min_img_dim=(130, 70)))
        artificial_transform.insert(1, transforms.RandomCrop(
            (60, 120), padding_mode="reflect"))

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
                    model = MUNIT.load_from_checkpoint(path)
                    model.eval()
                    model.cuda()

                    # paths
                    try1_path = os.path.join(
                        results_dir, f"artificial_content_random_style={epoch}.png")
                    try2_path = os.path.join(
                        results_dir, f"real_content_artificial_style={epoch}.png")
                    try3_path = os.path.join(
                        results_dir, f"real_content_random_style={epoch}.png")

                    if img_real is None or img_fake is None:
                        img_dim = model.hparams.img_dim
                        # transform1 = transforms.Compose([transforms.Resize(img_dim[1:]),
                        #                                 transforms.ToTensor(),
                        #                                 transforms.RandomHorizontalFlip(
                        #                                     p=0.5),
                        #                                 transforms.RandomVerticalFlip(
                        #                                     p=0.5),
                        #                                 transforms.ColorJitter(
                        #                                     hue=[-0.1, 0.1]),
                        #                                 transforms.Normalize([0.5], [0.5])])
                        transform1 = transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                                         DynamicPad(min_img_dim=(110, 60),
                                                                    padding_mode="edge"),
                                                         transforms.RandomCrop(
                                                             (55, 105), padding_mode="reflect"),
                                                         transforms.Resize(
                                                             img_dim[1:]),
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.RandomVerticalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.5], [0.5])])

                        if img_dim != (32, 64):
                            artificial_transform[2 if args.use_dynamic_padding else 0] = transforms.Resize(
                                img_dim[1:])
                        # transform2 = transforms.Compose(artificial_transform)
                        transform2 = transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                                         DynamicPad(min_img_dim=(130, 70),
                                                                    padding_mode="constant", padding_value=125),
                                                         transforms.RandomRotation(
                                                             degrees=5, resample=PIL.Image.NEAREST, fill=125),
                                                         transforms.RandomCrop(
                                                             (60, 120), padding_mode="reflect"),
                                                         transforms.Resize(
                                                             img_dim[1:]),
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.RandomVerticalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             [0.5], [0.5]),
                                                         AddNoise(alpha=0.07)])

                        datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                                              data_dir2=artificial_dir,
                                                              transform=transform1,
                                                              transform2=transform2,
                                                              img_size=img_dim[1:],
                                                              batch_size=n_samples)
                        datasetmodule.setup()
                        dataloader = datasetmodule.train_dataloader()
                        img_real, img_fake = next(iter(dataloader))

                    enc1 = model.generator.encoder1
                    dec1 = model.generator.decoder1

                    enc2 = model.generator.encoder2
                    dec2 = model.generator.decoder2

                    # TRY2 : real_content_artificial_style
                    img_styles = img_fake.to(model.device)
                    img_contents = img_real.to(model.device)

                    _, styles = enc1(img_styles)
                    contents, _ = enc2(img_contents)

                    grid = [th.ones(img_dim, device="cpu")] + \
                        [img_style.detach().cpu().clone()
                         for img_style in img_styles]

                    # style mixing done here
                    for i, content in enumerate(contents):
                        tmp = th.cat(
                            [th.unsqueeze(content, 0) for _ in range(n_samples)], dim=0)
                        imgs = dec1(tmp.to(model.device),
                                    styles.to(model.device))

                        grid.append(img_contents[i].detach().cpu().clone())
                        grid += [img.detach().cpu().clone() for img in imgs]

                    # normalize for nicer look
                    for i in range(len(grid)):
                        grid[i] = (grid[i] + 1) / 2

                    # save
                    grid = torchvision.utils.make_grid(
                        grid, nrow=n_samples + 1, pad_value=1, normalize=False)
                    torchvision.utils.save_image(grid, try2_path)

                    # TRY1 : artificial_content_random_style
                    img_contents = img_fake.to(model.device)

                    contents, _ = enc1(img_contents)
                    if random_styles_fake is None:
                        random_styles_fake = th.normal(0, 1, size=(
                            n_samples, model.generator.style_dim), device=model.device)

                    grid = []
                    # random styles
                    for i, content in enumerate(contents):
                        tmp = th.cat(
                            [th.unsqueeze(content, 0) for _ in range(n_samples)], dim=0)
                        imgs = dec1(tmp.to(model.device),
                                    random_styles_fake.to(model.device))

                        grid.append(img_contents[i].detach().cpu().clone())
                        grid += [img.detach().cpu().clone() for img in imgs]

                    # normalize for nicer look
                    for i in range(len(grid)):
                        grid[i] = (grid[i] + 1) / 2

                    # save
                    grid = torchvision.utils.make_grid(
                        grid, nrow=n_samples + 1, pad_value=1, normalize=False)
                    torchvision.utils.save_image(grid, try1_path)

                    # TRY3 : real_content_random_style
                    img_contents = img_real.to(model.device)

                    contents, _ = enc2(img_contents)
                    if random_styles_real is None:
                        random_styles_real = th.normal(0, 1, size=(
                            n_samples, model.generator.style_dim), device=model.device)

                    grid = []
                    # random styles
                    for i, content in enumerate(contents):
                        tmp = th.cat(
                            [th.unsqueeze(content, 0) for _ in range(n_samples)], dim=0)
                        imgs = dec2(tmp.to(model.device),
                                    random_styles_real.to(model.device))

                        grid.append(img_contents[i].detach().cpu().clone())
                        grid += [img.detach().cpu().clone() for img in imgs]

                    # normalize for nicer look
                    for i in range(len(grid)):
                        grid[i] = (grid[i] + 1) / 2

                    # save
                    grid = torchvision.utils.make_grid(
                        grid, nrow=n_samples + 1, pad_value=1, normalize=False)
                    torchvision.utils.save_image(grid, try3_path)
