import os
from shutil import copyfile

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *

# EXPERIMENTS
# --------------------------------
# version 0: (A) baseline
# version 1: (B) + padding reflect
# version 2: (C) + lr scheduler

generator_params = [
    # padding zero
    {"n_layers": 4,
     "init_channels": 512,
     "bn_mode": "default",
     "act": "leakyrelu",
     "last_layer_kernel_size": 3},
    # padding reflect
    {"n_layers": 4,
     "init_channels": 512,
     "padding_mode": "reflect",
     "bn_mode": "default",
     "act": "leakyrelu",
     "last_layer_kernel_size": 3},
    {"n_layers": 4,
     "init_channels": 512,
     "padding_mode": "reflect",
     "bn_mode": "default",
     "act": "leakyrelu",
     "last_layer_kernel_size": 3}
]
discriminator_params = [
    # padding zero
    {"base_channels": 64,
     "n_layers": 4,
     "bn_mode": "default"},
    # padding reflect
    {"base_channels": 64,
     "padding_mode": "reflect",
     "n_layers": 4,
     "bn_mode": "default"},
    {"base_channels": 64,
     "padding_mode": "reflect",
     "n_layers": 4,
     "bn_mode": "default"}
]

use_lr_schedulers = [False, False, True]

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 1000
interval = 25

data_dir = "/scratch/s7hialtu/potsdam_cars_corrected"
results_dir = "/scratch/s7hialtu/dcgan_experiments"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars_corrected"
    results_dir = "logs"

# DATA AUG FOR
transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(
                                    hue=[-0.1, 0.1], contrast=[1, 1.25]),
                                transforms.Normalize([0.5], [0.5])])

for generator_param, discriminator_param, use_lr_scheduler in zip(generator_params, discriminator_params, use_lr_schedulers):

    model = GAN(img_dim,
                discriminator_params=discriminator_param,
                disc_model="basic",
                generator_params=generator_param,
                gen_model="basic",
                fid_interval=interval,
                use_lr_scheduler=use_lr_scheduler)

    potsdam = PostdamCarsDataModule(data_dir,
                                    img_size=img_dim[1:],
                                    batch_size=batch_size,
                                    transform=transform)
    potsdam.setup()

    callbacks = [
        TensorboardGeneratorSampler(
            epoch_interval=interval, num_samples=batch_size, normalize=True),
        LatentDimInterpolator(
            interpolate_epoch_interval=interval, num_samples=10),
        ShowWeights(),
        ModelCheckpoint(period=interval, save_top_k=-1, filename="{epoch}"),
        EarlyStopping(monitor="fid", patience=10*interval, mode="min"),
        MyEarlyStopping(300, threshold=5, monitor="fid", mode="min")
    ]

    # Apparently Trainer has logger by default
    trainer = pl.Trainer(default_root_dir=results_dir,
                         gpus=1,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=20)
    try:
        trainer.fit(model, datamodule=potsdam)
    except KeyboardInterrupt:
        pass

file_name = os.path.basename(__file__)
copyfile(os.path.join("experiment", file_name),
         os.path.join(results_dir, file_name))
