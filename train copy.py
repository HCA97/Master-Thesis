import torchvision.utils as ut
import os
from shutil import copyfile

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
import PIL

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *

# POTSDAM CARS
generator_params = {"n_layers": 2,
                    "base_channels": 32, "padding_mode": "reflect"}
discriminator_params = {"base_channels": 64, "n_res": 1, "n_layers": 4,
                        "kernel_size": 3, "padding_mode": "reflect"}

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 1000
interval = 25

data_dir1 = "/scratch/s7hialtu/potsdam_cars_all"
data_dir2 = "/scratch/s7hialtu/artificial_cars"
results_dir = "/scratch/s7hialtu/munit_modified"

if not os.path.isdir(data_dir1):
    data_dir1 = "../potsdam_data/potsdam_cars_val"
    data_dir2 = "../potsdam_data/artificial_cars_small"
    results_dir = "logs"

# DATA AUG FOR
transform1 = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(hue=[-0.1, 0.1]),
                                transforms.Normalize([0.5], [0.5])])
transform2 = transforms.Compose([transforms.Resize((35, 67)),
                                 transforms.RandomCrop((32, 64)),
                                 transforms.ColorJitter(
                                     contrast=[1, 1.2], hue=[-0.2, 0.2]),
                                 transforms.RandomRotation(
                                     degrees=10, resample=PIL.Image.BILINEAR),
                                 transforms.RandomPerspective(
                                     distortion_scale=0.2),
                                 transforms.GaussianBlur(3),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

model = MUNIT(img_dim,
              discriminator_params=discriminator_params,
              generator_params=generator_params,
              gen_init="kaiming",
              learning_rate_gen=0.0001,
              learning_rate_disc=0.0004,
              use_lr_scheduler=True,
              fid_interval=interval,
              l4=0,
              use_lpips=False)
# print(model.generator)

# print(model.discriminator)

# assert False
potsdam = PostdamCarsDataModule(
    data_dir1, img_size=img_dim[1:], batch_size=batch_size,
    data_dir2=data_dir2, transform=transform1, transform2=transform2)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=interval, num_samples=batch_size, normalize=True),
    LatentDimInterpolator(
        interpolate_epoch_interval=interval, num_samples=10),
    ModelCheckpoint(period=interval, save_top_k=-1, filename="{epoch}"),
    # EarlyStopping(monitor="fid", patience=10*interval, mode="min"),
    MUNITCallback(epoch_interval=interval, n_samples=10),
    Pix2PixCallback(epoch_interval=interval),
    ShowWeights(),
    # MyEarlyStopping(300, threshold=5, monitor="fid", mode="min")
]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                     callbacks=callbacks, progress_bar_refresh_rate=1)
try:
    trainer.fit(model, datamodule=potsdam)
except KeyboardInterrupt:
    pass

file_name = os.path.basename(__file__)
copyfile(file_name,
         os.path.join(results_dir, file_name))