import os
import argparse

import tqdm
import torch as th
import torchvision
from torchvision import transforms

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

from scripts import *


def eval_on_version(checkpoint_dir: str,
                    results_dir: str,
                    potsdam_dir: str = "../potsdam_data/potsdam_cars",
                    interval: int = 25,
                    img_dim: tuple = (32, 64),
                    truncation: float = 1,
                    use_bn: bool = False,
                    layer_idx: int = 33,
                    skip_train_fid: bool = False,
                    skip_fid: bool = False,
                    data_aug: int = -1,
                    mode: str = "dcgan"):

    # other arguments
    max_epochs = 1000
    num_samples = 64
    steps = 5
    brightness = [1, 1.1]
    contrast = [1, 1.25]
    hue = [-0.1, 0.1]
    fid_samples = 1024

    checkpoints = [
        i-1 for i in range(interval, max_epochs, interval)] + [max_epochs-1]
    checkpoint_path = os.path.join(checkpoint_dir, "epoch={}.ckpt")

    os.makedirs(results_dir, exist_ok=True)

    fake_imgs = []
    inter_imgs = []

    # generate images from same latent points
    interpolate = LatentDimInterpolator(num_samples=10, steps=steps)

    # fid score stuff
    potsdam_dataset = PostdamCarsDataModule(
        potsdam_dir, transform=DATA_AUGS[data_aug], img_size=img_dim, batch_size=fid_samples)
    potsdam_dataset.setup()
    dataloader = potsdam_dataset.train_dataloader()

    if len(dataloader.dataset) < fid_samples and data_aug == 0:
        raise RuntimeError(
            f"{potsdam_dir} dataset size is too small, enable data augmentation")

    times = int(np.ceil(fid_samples / len(dataloader.dataset)))
    imgs1, imgs2 = [], []
    for _ in range(times):
        imgs2.append(next(iter(dataloader))[0])
        imgs1.append(next(iter(dataloader))[0])
    imgs1 = th.cat(imgs1, dim=0)
    imgs2 = th.cat(imgs2, dim=0)

    if not skip_train_fid and not skip_fid:
        fid = fid_score(imgs1, imgs2, device="cuda:0",
                        layer_idx=layer_idx, use_bn=use_bn)
        print(f"FID score of training set is {fid}.")

    z_fid = None
    fid_scores = []
    # lpips_scores = []
    xticks = []

    with th.no_grad():
        for i in tqdm.tqdm(checkpoints):
            path = checkpoint_path.format(i)
            if os.path.exists(path):
                xticks.append(i)

                model = GAN.load_from_checkpoint(path)
                model.eval()
                model.cuda()

                img_name = os.path.join(results_dir, f"images_epoch={i}.png")
                inter_name = os.path.join(
                    results_dir, f"interpolation_epoch={i}.png")

                # compute FID score
                if z_fid is None:
                    z_fid = th.normal(
                        0, truncation, (fid_samples, model.generator.latent_dim, 1, 1), device=model.device)
                imgs1 = model(z_fid)

                if not skip_fid:
                    fid = fid_score(imgs1, imgs2, device="cuda:0",
                                    layer_idx=layer_idx, use_bn=use_bn)
                    fid_scores.append(fid)
                    print(f"Epoch {i} - FID {fid}")

                # save the images
                for k, img in enumerate(imgs1):
                    img_folder = os.path.join(
                        results_dir, "images", f"epoch={i}")
                    os.makedirs(img_folder, exist_ok=True)
                    img = img.detach().cpu().numpy().transpose(1, 2, 0)
                    img = (img + 1) / 2
                    cv2.imwrite(os.path.join(img_folder, f"{k}.png"), cv2.cvtColor(
                        255*img, cv2.COLOR_RGB2BGR))

                # # compute PPL Score
                # ppl = perceptual_path_length(
                #     model.generator, n_samples=2048, truncation=truncation, device=model.device, epsilon=1e-2)
                # lpips_scores.append(ppl)

                # generate examples
                grid = torchvision.utils.make_grid(
                    imgs1[:num_samples], nrow=8, normalize=True)
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
                    model, truncation=truncation)
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

    # save fid scores
    if not skip_fid:
        np.save(os.path.join(results_dir, "scores.npy"),
                {"epochs": xticks, "fid": fid_scores})
        # np.save(os.path.join(results_dir, "scores.npy"),
        #         {"epochs": xticks, "fid": fid_scores, "ppl": lpips_scores})

        plt.figure()
        plt.plot(xticks, fid_scores, marker="*")
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("FID Score", fontsize=18)
        plt.title("FID Score", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "fid_score.png"))

    # plt.figure()
    # plt.plot(xticks, lpips_scores, marker="*")
    # plt.xlabel("Epochs", fontsize=18)
    # plt.ylabel("PPL Score", fontsize=18)
    # plt.title("PPL Score", fontsize=20)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, "ppl_score.png"))

    # save them as gif
    imageio.mimsave(os.path.join(results_dir, "images.gif"),
                    fake_imgs, fps=1)

    imageio.mimsave(os.path.join(results_dir, "interpolation.gif"),
                    inter_imgs, fps=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluation script for DCGAN and PIX2PIX")
    parser.add_argument("experiment_dir", help="experiment folder")
    parser.add_argument(
        "mode", choices=["dcgan", "pix2pix"], help="normal gan or pix2pix")
    parser.add_argument(
        "--potsdam_dir", default="../potsdam_data/potsdam_cars", help="path to potsdam cars")
    parser.add_argument("--interval", default=25, type=int,
                        help="interval between two epochs")
    parser.add_argument("--not_use_bn", action="store_false",
                        help="not use vgg with bn", dest="use_bn")
    parser.add_argument("--skip_train_fid", action="store_true",
                        help="skip fid of training dataset", dest="skip_train_fid")
    parser.add_argument("--skip_fid", action="store_true",
                        help="skip fid computation", dest="skip_fid")
    parser.add_argument("--width", type=int, default=64, help="image width")
    parser.add_argument("--height", type=int, default=32, help="image height")
    parser.add_argument("--truncation", type=float, default=1,
                        help="latent space truncation")
    parser.add_argument("--data_aug", type=int, default=0)
    parser.add_argument("--result_dir", default="results",
                        help="results dir name")
    parser.set_defaults(use_bn=True)
    parser.set_defaults(skip_train_fid=False)
    parser.set_defaults(skip_fid=False)
    args = parser.parse_args()

    # path
    experiment_dir = args.experiment_dir
    potsdam_dir = args.potsdam_dir
    results_dir_name = args.result_dir

    # other
    img_dim = (args.height, args.width)
    truncation = args.truncation
    data_aug = args.data_aug
    layer_idx = 33 if args.use_bn else 24
    interval = args.interval
    mode = args.mode

    # flags
    use_bn = args.use_bn
    skip_fid = args.skip_fid
    skip_train_fid = args.skip_train_fid

    for version in os.listdir(os.path.join(experiment_dir, "lightning_logs")):

        print("Version ", version)
        checkpoint_dir = os.path.join(
            experiment_dir, "lightning_logs", version, "checkpoints")
        results_dir = os.path.join(experiment_dir, version, results_dir_name)

        eval_on_version(checkpoint_dir, results_dir,
                        potsdam_dir,
                        interval,
                        img_dim,
                        truncation,
                        use_bn,
                        layer_idx,
                        skip_train_fid,
                        skip_fid,
                        data_aug,
                        mode)
