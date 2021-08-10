import torchvision.utils as ut
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

# pl.utilities.seed.seed_everything(seed=0)

# POTSDAM CARS
generator_params = {"n_layers": 4,
                    "base_channels": 32,
                    "padding_mode": "reflect"}
discriminator_params = {
    "n_res": 1,
    "disc_parameters": {"base_channels": 64,
                        "padding_mode": "reflect",
                        "use_sigmoid": False,
                        "use_dropout": False,
                        "use_spectral_norm": True,
                        "use_instance_norm": True,
                        "n_layers": 4
                        }
}

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 500
interval = 25

data_dir1 = "/scratch/s7hialtu/potsdam_cars_all"
data_dir2 = "/scratch/s7hialtu/training_tightcanvas_graybackground"
results_dir = "/scratch/s7hialtu/munit_artificial2real_graybackground_dynamic_padding_noise"

if not os.path.isdir(data_dir1):
    data_dir1 = "../potsdam_data/potsdam_cars_all"
    data_dir2 = "../potsdam_data/cem-v0/v2/training_tightcanvas_graybackground"
    results_dir = "logs"

# DATA AUG FOR
transform1 = transforms.Compose([transforms.ColorJitter(hue=[-0.1, 0.1]),
                                 DynamicPad(min_img_dim=(110, 60),
                                            padding_mode="edge"),
                                 transforms.RandomCrop(
                                     (55, 105), padding_mode="reflect"),
                                 transforms.Resize((32, 64)),
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
                                 transforms.Resize((32, 64)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5]),
                                 AddNoise(alpha=0.07)])


model = MUNIT(img_dim,
              discriminator_params=discriminator_params,
              generator_params=generator_params,
              gen_init="kaiming",
              disc_model="basic",
              learning_rate_gen=0.0001,
              learning_rate_disc=0.0004,
              weight_decay_disc=1e-3,
              weight_decay_gen=1e-3,
              use_lr_scheduler=True,
              l4=10,
              fid_interval=interval,
              use_lpips=False)

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
    MUNITCallback(epoch_interval=interval),
    Pix2PixCallback(epoch_interval=interval),
    ShowWeights()
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
