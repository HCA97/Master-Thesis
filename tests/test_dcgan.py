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
generator_params = {"n_layers": 5, "init_channels": 1024}
discriminator_params = {"base_channels": 32,
                        "n_layers": 5, "heat_map_layer": 2, "heat_map": True}

img_dim = (3, 64, 128)
batch_size = 124
max_epochs = 5


data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_heatmap"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

model = GAN(img_dim, discriminator_params=discriminator_params, fid_interval=1, use_buffer=True,
            generator_params=generator_params, gen_model="basic", use_lr_scheduler=True, use_symmetry_loss=True)

potsdam = PostdamCarsDataModule(
    data_dir, data_dir2=data_dir, img_size=img_dim[1:], batch_size=batch_size)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=1, num_samples=batch_size, normalize=True),
    LatentDimInterpolator(interpolate_epoch_interval=1, num_samples=10),
    ModelCheckpoint(period=1, save_top_k=-1, filename="{epoch}"),
    EarlyStopping(monitor="fid", patience=5, mode="min"),
    Pix2PixCallback(),
    ShowWeights(),
    MyEarlyStopping(5, threshold=5)]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=potsdam)
