# from argparse import ArgumentParser
import os

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *


# MNIST TEST EXAMPLE
# model = GAN((1, 32, 32))
# mnist = MNISTDataModule(batch_size=128)
# mnist.setup()

# callbacks = [
#     TensorboardGeneratorSampler(num_samples=64),
#     LatentDimInterpolator(interpolate_epoch_interval=1)]

# # Apparently Trainer has logger by default
# trainer = pl.Trainer(default_root_dir="logs", gpus=1, max_epochs=1,
#                      callbacks=callbacks, progress_bar_refresh_rate=20)
# trainer.fit(model, datamodule=mnist)


# POTSDAM CARS
img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 500
data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

model = GAN(img_dim)

potsdam = PostdamCarsDataModule(data_dir, batch_size=batch_size)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=5, num_samples=batch_size, normalize=True),
    LatentDimInterpolator(interpolate_epoch_interval=5, num_samples=10),
    ModelCheckpoint(period=5, save_top_k=-1, filename="{epoch}")
]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=potsdam)
