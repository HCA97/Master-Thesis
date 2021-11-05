import os
import argparse

import tqdm
import torch as th
import torchvision
from torchvision import transforms

import cv2
import numpy as np

from scripts import *


def foo(img, s, encoder1, decoder2):
    c1, _ = encoder1(img)
    return decoder2(c1, s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Generate samples from each checkpoint for MUNIT")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument(
        "--potsdam_dir", default="../potsdam_data/potsdam_cars_corrected", help="path to potsdam cars")
    parser.add_argument(
        "--artificial_dir", default="../potsdam_data/cem-v0/v2/training_tightcanvas_graybackground", help="path to artificial cars")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--n_samples", default=1024,
                        type=int, help="number of fake images")
    parser.add_argument("--results_name", default="results",
                        help="results name")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--data_aug", type=int, default=0)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    potsdam_dir = args.potsdam_dir
    artificial_dir = args.artificial_dir

    n_samples = args.n_samples
    interval = args.interval

    results_name = args.results_name
    wanted_version = args.version
    data_aug = args.data_aug

    # other
    gen_in = None
    img_real = None
    artificial_random_styles = None
    real_random_styles = None

    n_samples_show = 60
    n_styles = 10
    n_contents = 10
    n_random_styles = 10

    max_epochs = 1000
    bs = 512

    s = th.normal(0, 1, (n_samples, 8))

    transforms_ = [
        (  # Dynamic Pad. + Gaus Noise + Color Jitter + Horizontal/Vertical Flip
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                DynamicPad(min_img_dim=(110, 60),
                                           padding_mode="edge"),
                                transforms.RandomCrop(
                                    (55, 105), padding_mode="reflect"),
                                transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                DynamicPad(min_img_dim=(130, 70),
                                           padding_mode="constant", padding_value=125),
                                transforms.RandomRotation(
                                    degrees=5, resample=PIL.Image.NEAREST, fill=125),
                                transforms.RandomCrop(
                                    (60, 120), padding_mode="reflect"),
                                transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                                AddNoise(alpha=0.07)])
        ),
        (  # Horizontal/Vertical Flip
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
        ),
        (  # Color Jitter + Gaus Noise + Horizontal/Vertical Flip
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                transforms.Resize((32, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                                AddNoise(alpha=0.07)])
        ),
        (  # Dynamic Pad. + Gaus Noise + Color Jitter + Horizontal/Vertical Flip + Larger Image
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                DynamicPad(min_img_dim=(110, 60),
                                           padding_mode="edge"),
                                transforms.RandomCrop(
                                    (55, 105), padding_mode="reflect"),
                                transforms.Resize((40, 80)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                DynamicPad(min_img_dim=(130, 70),
                                           padding_mode="constant", padding_value=125),
                                transforms.RandomRotation(
                                    degrees=5, resample=PIL.Image.NEAREST, fill=125),
                                transforms.RandomCrop(
                                    (60, 120), padding_mode="reflect"),
                                transforms.Resize((40, 80)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                                AddNoise(alpha=0.07)])

        )
    ]

    if n_samples % bs != 0:
        raise RuntimeError(
            f"N samples ({n_samples}) must be divisible by batch size ({bs})")

    # checkpoints
    checkpoints = [
        i-1 for i in range(interval, max_epochs, interval)] + [max_epochs-1]

    for version in os.listdir(os.path.join(experiment_dir, "lightning_logs")):

        print(f"Version : {version}")
        if version == f"version_{wanted_version}":

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

                        encoder1 = model.generator.encoder1
                        encoder2 = model.generator.encoder2
                        decoder1 = model.generator.decoder1
                        decoder2 = model.generator.decoder2

                        # paths
                        img_folder = os.path.join(
                            results_dir, "images", f"epoch={epoch}")
                        samples_path = os.path.join(
                            results_dir, f"epoch={epoch}.png")

                        real2artificial_path = os.path.join(
                            results_dir, "real2artificial", f"real2artificial={epoch}.png")
                        style_path = os.path.join(
                            results_dir, "style_mix", f"style_mix={epoch}.png")
                        different_styles = os.path.join(
                            results_dir, "style_samples", f"styles_sample={epoch}.png")

                        # makedirs
                        os.makedirs(os.path.split(real2artificial_path)
                                    [0], exist_ok=True)
                        os.makedirs(os.path.split(style_path)
                                    [0], exist_ok=True)
                        os.makedirs(os.path.split(different_styles)
                                    [0], exist_ok=True)

                        # this is the last thing we will do before we move to the next checkpoint
                        if os.path.exists(different_styles):
                            continue

                        os.makedirs(img_folder, exist_ok=True)

                        if gen_in is None:
                            transform1, transform2 = transforms_[data_aug]

                            datasetmodule = PostdamCarsDataModule(potsdam_dir,
                                                                  data_dir2=artificial_dir,
                                                                  transform=transform1,
                                                                  transform2=transform2,
                                                                  batch_size=n_samples)
                            datasetmodule.setup()
                            dataloader = datasetmodule.train_dataloader()

                            # in case dataset smaller than n_samples
                            times = int(
                                np.ceil(n_samples / len(dataloader.dataset)))
                            gen_in = th.cat([next(iter(dataloader))[1]
                                            for _ in range(times)], dim=0)

                            # since we will only use 10 samples we don't care how small is this
                            img_real = next(iter(dataloader))[0]

                        # need to do it in batches for memory stuff
                        fake_imgs = []
                        for i in range(n_samples // bs):
                            si = i * bs
                            ei = min((i+1)*bs, n_samples)

                            artificial_cars = gen_in[si:ei].clone().to(
                                model.device)
                            random_styles = s[si:ei].clone().to(model.device)

                            artificial_content = encoder1(artificial_cars)[0]
                            imgs = decoder2(artificial_content, random_styles)

                            fake_imgs.append(imgs.detach().cpu().clone())
                        fake_imgs = th.cat(fake_imgs, dim=0)

                        # save them
                        for k, img in enumerate(fake_imgs):
                            img = img.detach().cpu().numpy().transpose(1, 2, 0)
                            img = (img + 1) / 2
                            cv2.imwrite(os.path.join(img_folder, f"{k}.png"), cv2.cvtColor(
                                255*img, cv2.COLOR_RGB2BGR))

                        # show small sample for visualization
                        grid = torchvision.utils.make_grid(
                            fake_imgs[:n_samples_show], nrow=10, normalize=True, pad_value=1)
                        torchvision.utils.save_image(grid, samples_path)

                        #
                        # EXTRA STUFF
                        #

                        # ------------------
                        # real2artificial
                        # ------------------
                        img_contents = img_real[:n_contents].to(model.device)

                        real_contents, _ = encoder2(img_contents)
                        if real_random_styles is None:
                            real_random_styles = th.normal(0, 1, size=(
                                n_random_styles, model.generator.style_dim), device=model.device)

                        grid = []
                        for i, content in enumerate(real_contents):
                            # for each content create n_random_styles
                            row = th.cat(
                                [th.unsqueeze(content, 0) for _ in range(n_random_styles)], dim=0)
                            imgs = decoder1(row.to(model.device),
                                            real_random_styles)

                            # a row consists of content image and its derivatives
                            grid.append(img_contents[i].detach().cpu().clone())
                            grid += [img.detach().cpu().clone()
                                     for img in imgs]

                        # save everyting
                        grid = torchvision.utils.make_grid(
                            grid, nrow=n_random_styles + 1, pad_value=1, normalize=True, range=(-1, 1))
                        torchvision.utils.save_image(
                            grid, real2artificial_path)

                        # ------------------
                        # style mixing
                        # ------------------

                        img_styles = img_real[:n_styles].to(model.device)
                        img_contents = gen_in[:n_contents].to(model.device)

                        contents, _ = encoder1(img_contents)
                        _, styles = encoder2(img_styles)

                        # image styles will be the first row
                        grid = [th.ones(img_real.shape[1:], device="cpu")] + \
                            [img_style.detach().cpu().clone()
                             for img_style in img_styles]

                        for i, content in enumerate(contents):
                            # for each content create n_random_styles
                            row = th.cat(
                                [th.unsqueeze(content, 0) for _ in range(n_styles)], dim=0)
                            imgs = decoder2(row.to(model.device),
                                            styles.to(model.device))

                            # a row consists of content image and its style mixing
                            grid.append(img_contents[i].detach().cpu().clone())
                            grid += [img.detach().cpu().clone()
                                     for img in imgs]

                        # save everyting
                        grid = torchvision.utils.make_grid(
                            grid, nrow=n_styles + 1, pad_value=1, normalize=True, range=(-1, 1))
                        torchvision.utils.save_image(grid, style_path)

                        # ------------------
                        # different style samples
                        # ------------------

                        img_contents = gen_in[:n_contents].to(model.device)

                        artificial_contents, _ = encoder1(img_contents)
                        if artificial_random_styles is None:
                            artificial_random_styles = th.normal(0, 1, size=(
                                n_random_styles, model.generator.style_dim), device=model.device)

                        grid = []
                        for i, content in enumerate(artificial_contents):
                            # for each content create n_random_styles
                            row = th.cat(
                                [th.unsqueeze(content, 0) for _ in range(n_random_styles)], dim=0)
                            imgs = decoder2(row.to(model.device),
                                            artificial_random_styles)

                            # a row consists of content image and its derivations
                            grid.append(img_contents[i].detach().cpu().clone())
                            grid += [img.detach().cpu().clone()
                                     for img in imgs]

                        # save everyting
                        grid = torchvision.utils.make_grid(
                            grid, nrow=n_random_styles + 1, pad_value=1, normalize=True, range=(-1, 1))
                        torchvision.utils.save_image(grid, different_styles)
