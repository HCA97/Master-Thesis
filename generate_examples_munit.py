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
        "--potsdam_dir", default="../potsdam_data/potsdam_cars_all", help="path to potsdam cars")
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/artificial_cars", help="path to artificial cars")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--n_samples", default=1024,
                        type=int, help="number of fake images")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    potsdam_dir = args.potsdam_dir
    artificial_dir = args.artificial_dir
    n_samples = args.n_samples
    interval = args.interval
    results_name = args.results_name

    # other
    n_samples_show = 64
    gen_in = None

    img_real = None
    random_styles = None
    n_styles = 10
    n_contents = 10
    n_random_styles = 10

    max_epochs = 1000
    bs = 512

    if n_samples % bs != 0:
        raise RuntimeError(
            f"N samples ({n_samples}) must be divisible by batch size ({bs})")
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
                    img_folder = os.path.join(
                        results_dir, "images", f"epoch={epoch}")
                    style_path = os.path.join(
                        results_dir, f"style_mix={epoch}.png")
                    samples_path = os.path.join(
                        results_dir, f"epoch={epoch}.png")
                    different_styles = os.path.join(
                        results_dir, f"styles_sample={epoch}.png")

                    # speed up the computation
                    if os.path.exists(different_styles):
                        continue

                    os.makedirs(img_folder, exist_ok=True)

                    if gen_in is None:
                        img_dim = model.hparams.img_dim
                        transform1 = transforms.Compose([transforms.Resize(img_dim[1:]),
                                                        transforms.ToTensor(),
                                                        transforms.RandomHorizontalFlip(
                                                            p=0.5),
                                                        transforms.RandomVerticalFlip(
                                                            p=0.5),
                                                        transforms.ColorJitter(
                                                            hue=[-0.1, 0.1]),
                                                        transforms.Normalize([0.5], [0.5])])
                        transform2 = transforms.Compose([transforms.Resize(img_dim[1:]),
                                                        transforms.ToTensor(),
                                                        transforms.RandomHorizontalFlip(
                                                            p=0.5),
                                                        transforms.RandomVerticalFlip(
                                                            p=0.5),
                                                        #  transforms.ColorJitter(
                                                         #     hue=[-0.1, 0.5]),
                                                         transforms.Normalize([0.5], [0.5])])

                        datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                                              data_dir2=artificial_dir,
                                                              transform=transform1,
                                                              transform2=transform2,
                                                              img_size=img_dim[1:],
                                                              batch_size=n_samples)
                        datasetmodule.setup()
                        dataloader = datasetmodule.train_dataloader()

                        times = int(
                            np.ceil(n_samples / len(dataloader.dataset)))

                        gen_in = th.cat([next(iter(dataloader))[1]
                                        for _ in range(times)], dim=0)
                        img_real = next(iter(dataloader))[0]

                    # fake imgs need to do it in batches for memory stuff
                    fake_imgs = []
                    for i in range(n_samples // bs):
                        si = i * bs
                        ei = min((i+1)*bs, n_samples)
                        sample = gen_in[si:ei].clone().to(model.device)
                        fake_imgs.append(
                            model(sample, th.zeros_like(sample), inference=True)[0].detach().cpu().clone())
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

                    # style mixing
                    enc1 = model.generator.encoder1
                    enc2 = model.generator.encoder2
                    dec2 = model.generator.decoder2

                    img_styles = img_real[:n_styles].to(model.device)
                    img_contents = gen_in[:n_contents].to(model.device)

                    contents, _ = enc1(img_contents)
                    _, styles = enc2(img_styles)

                    grid = [th.ones(img_dim, device="cpu")] + \
                        [img_style.detach().cpu().clone()
                         for img_style in img_styles]
                    for i, content in enumerate(contents):
                        tmp = th.cat(
                            [th.unsqueeze(content, 0) for _ in range(n_styles)], dim=0)
                        imgs = dec2(tmp.to(model.device),
                                    styles.to(model.device))

                        grid.append(img_contents[i].detach().cpu().clone())
                        grid += [img.detach().cpu().clone() for img in imgs]

                    for i in range(len(grid)):
                        grid[i] = (grid[i] + 1) / 2

                    grid = torchvision.utils.make_grid(
                        grid, nrow=n_styles + 1, pad_value=1, normalize=False)
                    torchvision.utils.save_image(grid, style_path)

                    # samples
                    enc1 = model.generator.encoder1
                    dec2 = model.generator.decoder2

                    img_contents = gen_in[:n_contents].to(model.device)

                    contents, _ = enc1(img_contents)
                    if random_styles is None:
                        random_styles = th.normal(0, 1, size=(
                            n_random_styles, model.generator.style_dim), device=model.device)

                    grid = []
                    for i, content in enumerate(contents):
                        tmp = th.cat(
                            [th.unsqueeze(content, 0) for _ in range(n_random_styles)], dim=0)
                        imgs = dec2(tmp.to(model.device),
                                    random_styles.to(model.device))

                        grid.append(img_contents[i].detach().cpu().clone())
                        grid += [img.detach().cpu().clone() for img in imgs]

                    for i in range(len(grid)):
                        grid[i] = (grid[i] + 1) / 2

                    grid = torchvision.utils.make_grid(
                        grid, nrow=n_random_styles + 1, pad_value=1, normalize=False)
                    torchvision.utils.save_image(grid, different_styles)
