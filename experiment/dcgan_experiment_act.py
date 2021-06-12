import os

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *

# POTSDAM CARS
generator_params = [{"n_layers": 4, "init_channels": 512, "act": "relu"},
                    {"n_layers": 4, "init_channels": 512, "act": "leakyrelu"}
                    ]
discriminator_params = {"base_channels": 32, "n_layers": 4}

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 1000
interval = 25

data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_act"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

for generator_param in generator_params:
    model = GAN(img_dim, discriminator_params=discriminator_params, fid_interval=interval,
                generator_params=generator_param, gen_model="basic")

    potsdam = PostdamCarsDataModule(
        data_dir, img_size=img_dim[1:], batch_size=batch_size)
    potsdam.setup()

    callbacks = [
        TensorboardGeneratorSampler(
            epoch_interval=interval, num_samples=batch_size, normalize=True),
        LatentDimInterpolator(
            interpolate_epoch_interval=interval, num_samples=10),
        ModelCheckpoint(period=interval, save_top_k=-1, filename="{epoch}"),
        EarlyStopping(monitor="fid", patience=10*interval, mode="min"),
        Pix2PixCallback(epoch_interval=interval),
        ShowWeights(),
        MyEarlyStopping(300, max_fid=5)
    ]

    # Apparently Trainer has logger by default
    trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                         callbacks=callbacks, progress_bar_refresh_rate=20)
    trainer.fit(model, datamodule=potsdam)
