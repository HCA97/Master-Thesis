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

# MODELS
generator_params = {"n_layers": 4, "init_channels": 512,
                    "bn_mode": "default", "use_spectral_norm": True}
discriminator_params = {"base_channels": 32, "n_layers": 4,
                        "bn_mode": "default", "use_spectral_norm": True}

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 1000
interval = 25

# DATA AUG
transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(hue=[-0.1, 0.1]),
                                transforms.Normalize([0.5], [0.5])])

# POTSDAM CARS
data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_spectral_norm"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"


model = GAN(img_dim, discriminator_params=discriminator_params, fid_interval=interval,
            generator_params=generator_params, gen_model="basic")

potsdam = PostdamCarsDataModule(
    data_dir, img_size=img_dim[1:], batch_size=batch_size, transform=transform)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=interval, num_samples=batch_size, normalize=True),
    LatentDimInterpolator(
        interpolate_epoch_interval=interval, num_samples=10),
    ModelCheckpoint(period=interval, save_top_k=-1, filename="{epoch}"),
    EarlyStopping(monitor="fid", patience=20*interval, mode="min"),
    Pix2PixCallback(epoch_interval=interval),
    ShowWeights(),
    MyEarlyStopping(300, threshold=5, monitor="fid", mode="min")
]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=potsdam)

file_name = os.path.basename(__file__)
copyfile(file_name,
         os.path.join(results_dir, file_name))
