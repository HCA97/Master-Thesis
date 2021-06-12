# from argparse import ArgumentParser
from shutil import copyfile
import os

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from scripts import *

# data augmentation
brightness = [1, 1.1]
contrast = [1, 1.25]
hue = [-0.1, 0.1]

# POTSDAM CARS
generator_params = {"n_layers": 4, "init_channels": 512}
discriminator_params = {"base_channels": 32, "n_layers": 4, "heat_map": False}
use_gp = False
img_dim = (3, 32, 64)

dataaug = [
    # horizontal flip only
    transforms.Compose([transforms.Resize(img_dim[1:]),
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.Normalize([0.5], [0.5])]),
    # horizontal flip + hue
    transforms.Compose([transforms.Resize(img_dim[1:]),
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(
                                                hue=hue)], p=1),
                        transforms.Normalize([0.5], [0.5])]),
    # horizontal flip + hue + contrast
    transforms.Compose([transforms.Resize(img_dim[1:]),
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(
                                                hue=hue, contrast=contrast)], p=1),
                        transforms.Normalize([0.5], [0.5])]),
    # horizontal flip + hue + contrast + brightness
    transforms.Compose([transforms.Resize(img_dim[1:]),
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(
                                                hue=hue, brightness=brightness, contrast=contrast)], p=1),
                        transforms.Normalize([0.5], [0.5])]),
]


batch_size = 64
max_epochs = 500
data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_bigger/dataaug"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

for transform in dataaug:
    # call backs
    callbacks = [
        TensorboardGeneratorSampler(
            epoch_interval=25, num_samples=batch_size, normalize=True),
        LatentDimInterpolator(interpolate_epoch_interval=25, num_samples=10),
        ModelCheckpoint(period=25,
                        save_top_k=-1, filename="{epoch}")
    ]

    # dataset
    potsdam = PostdamCarsDataModule(
        data_dir, img_size=img_dim[1:], batch_size=batch_size, transform=transform)
    potsdam.setup()

    # model
    model = GAN(img_dim, discriminator_params=discriminator_params,
                generator_params=generator_params, use_gp=use_gp)

    # training
    trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                         callbacks=callbacks, progress_bar_refresh_rate=20)
    trainer.fit(model, datamodule=potsdam)

file_name = os.path.basename(__file__)
copyfile(file_name, os.path.join(results_dir, file_name))
