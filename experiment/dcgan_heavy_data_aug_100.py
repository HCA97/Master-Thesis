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

# POTSDAM CARS
generator_param = {"n_layers": 4,
                   "init_channels": 512,
                   "bn_mode": "default",
                   "act": "leakyrelu",
                   "last_layer_kernel_size": 3}
discriminator_param = {"base_channels": 64,
                       "n_layers": 4,
                       "bn_mode": "default"}


img_dim = (3, 64, 128)
batch_size = 100
max_epochs = 18000
interval = 900

data_dir = "/scratch/s7hialtu/potsdam_cars_100"
results_dir = "/scratch/s7hialtu/dcgan_heavy_data_aug_100"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars_100"
    results_dir = "logs"

# DATA AUG FOR
transform = transforms.Compose([transforms.ColorJitter(hue=[-0.2, 0.2], contrast=(1, 1.2)),
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
model = GAN(img_dim,
            discriminator_params=discriminator_param,
            disc_model="basic",
            generator_params=generator_param,
            gen_model="basic",
            fid_interval=-1)

potsdam = PostdamCarsDataModule(data_dir,
                                img_size=img_dim[1:],
                                batch_size=batch_size,
                                transform=transform)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=interval, num_samples=64, normalize=True),
    LatentDimInterpolator(
        interpolate_epoch_interval=interval, num_samples=10),
    ShowWeights(),
    ModelCheckpoint(period=interval, save_top_k=-1, filename="{epoch}"),
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
