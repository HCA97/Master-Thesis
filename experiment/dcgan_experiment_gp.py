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
generator_params = {"n_layers": 4, "init_channels": 512}
discriminator_params = {"base_channels": 32, "n_layers": 4, "heat_map": False}
use_gp = True
alphas = [0.1, 1, 10]

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 500
data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_bigger/gp"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

potsdam = PostdamCarsDataModule(
    data_dir, img_size=img_dim[1:], batch_size=batch_size)
potsdam.setup()


for alpha in alphas:
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
        data_dir, img_size=img_dim[1:], batch_size=batch_size)
    potsdam.setup()

    # model
    model = GAN(img_dim, discriminator_params=discriminator_params,
                generator_params=generator_params, use_gp=use_gp, alpha=alpha)

    # training
    trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                         callbacks=callbacks, progress_bar_refresh_rate=20)
    try:
        trainer.fit(model, datamodule=potsdam)
    except KeyboardInterrupt:
        pass

file_name = os.path.basename(__file__)
copyfile(os.path.join("experiment", file_name),
         os.path.join(results_dir, file_name))
