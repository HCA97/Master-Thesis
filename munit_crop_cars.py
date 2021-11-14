import argparse
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tqdm
import torch as th
import torchvision
from torchvision import transforms

import cv2
import numpy as np

from scripts import *


save_path = "experiments/munit_version_2_larger_image_corrected_500/version_0/10000_cars_cropped_truncated"
path = "experiments/munit_version_2_larger_image_corrected_500/lightning_logs/version_0/checkpoints/epoch=499.ckpt"
artificial_dir = "../potsdam_data/cem-v0/v2/training_tightcanvas_graybackground"
idx = 0  # 6

# save_path = "experiments/munit_version_2_larger_image_corrected/version_1/10000_cars_cropped_truncated"
# path = "experiments/munit_version_2_larger_image_corrected_500/lightning_logs/version_0/checkpoints/epoch=899.ckpt"
# artificial_dir = "../potsdam_data/cem-v0/v2/training_tightcanvas_graybackground"
# idx = 6

img_dim = (3, 40, 80)
n_samples = 1000
bufferx = 3
buffery = 3


os.makedirs(save_path, exist_ok=True)


transform1 = transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                 DynamicPad(min_img_dim=(110, 60),
                                            padding_mode="edge"),
                                 transforms.RandomCrop(
                                     (55, 105), padding_mode="reflect"),
                                 transforms.Resize(img_dim[1:]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

transform2 = transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                 DynamicPad(min_img_dim=(130, 70),
                                            padding_mode="constant", padding_value=125),
                                 transforms.RandomRotation(
                                     degrees=5, resample=PIL.Image.NEAREST, fill=125),
                                 transforms.RandomCrop(
                                     (60, 120), padding_mode="reflect"),
                                 transforms.Resize(img_dim[1:]),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5]),
                                 AddNoise(alpha=0.07)])

datasetmodule = PostdamCarsDataModule(artificial_dir,
                                      transform=transform2,
                                      img_size=img_dim[1:],
                                      batch_size=n_samples)
datasetmodule.setup()
dataloader = datasetmodule.train_dataloader()

model = MUNIT.load_from_checkpoint(path)
model.eval()
model.cuda()

enc1 = model.generator.encoder1
dec1 = model.generator.decoder1

enc2 = model.generator.encoder2
dec2 = model.generator.decoder2

with th.no_grad():
    for i in tqdm.tqdm(range(10)):

        _, img_fake = next(iter(dataloader))
        img_contents = img_fake.to(model.device)
        random_styles = th.normal(0, 0.75, size=(
            n_samples, model.generator.style_dim), device=model.device)

        contents, _ = enc1(img_contents)
        imgs = dec2(contents, random_styles, False).detach().cpu().clone()
        before_rgbs = dec2(contents, random_styles,
                           True).detach().cpu().clone()

        for j in range(n_samples):

            img_realistic = imgs[j]
            before_rgb = before_rgbs[j]

            m = th.squeeze(before_rgb[idx]).numpy()
            if np.isclose(m.max(), 0):
                img_realistic = th.squeeze(
                    img_realistic).permute(1, 2, 0).numpy()
                img = img_realistic
                img = (img + 1) / 2
                cv2.imwrite(os.path.join(save_path, f"{i*n_samples + j}.png"), cv2.cvtColor(
                    255*img, cv2.COLOR_RGB2BGR))
            else:
                m /= m.max()
                m[m < 0.1] = 0
                m[m > 0.1] = 1

                r, c = np.where(m == 1)
                r_min, r_max = max(
                    0, r.min() - bufferx), min(img_dim[1], r.max() + bufferx)
                c_min, c_max = max(
                    0, c.min() - buffery), min(img_dim[2], c.max() + buffery)

                img_realistic = th.squeeze(
                    img_realistic).permute(1, 2, 0).numpy()
                img = img_realistic[r_min:r_max, c_min:c_max, :]
                img = (img + 1) / 2
                cv2.imwrite(os.path.join(save_path, f"{i*n_samples + j}.png"), cv2.cvtColor(
                    255*img, cv2.COLOR_RGB2BGR))
