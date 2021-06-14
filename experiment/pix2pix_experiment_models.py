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
generator_params = [("unet", {"n_layers": 4, "init_channels": 64, "act": "leakyrelu",
                    "bn_mode": "default", "n_blocks": 2, "reconstruction": True}),
                    ("refiner", {"n_layers": 4, "init_channels": 256, "act": "leakyrelu",
                                 "bn_mode": "default", "n_layers": 4, "reconstruction": True})]
discriminator_params = {"base_channels": 32,
                        "n_layers": 4, "bn_mode": "default"}
beta = 1e-4

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 1000
interval = 25

data_dir1 = "/scratch/s7hialtu/potsdam_cars"
data_dir2 = "/scratch/s7hialtu/artificial_cars"
results_dir = "/scratch/s7hialtu/pix2pix_unet_model"

if not os.path.isdir(data_dir1):
    data_dir1 = "../potsdam_data/potsdam_cars"
    data_dir2 = "../potsdam_data/artificial_cars"
    results_dir = "logs"

for gen_model, generator_param in generator_params:
    model = GAN(img_dim, discriminator_params=discriminator_params, fid_interval=interval,
                generator_params=generator_param, gen_model=gen_model, beta=beta)

    potsdam = PostdamCarsDataModule(
        data_dir1, img_size=img_dim[1:], batch_size=batch_size, data_dir2=data_dir2)
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
    try:
        trainer.fit(model, datamodule=potsdam)
    except KeyboardInterrupt:
        pass

file_name = os.path.basename(__file__)
copyfile(os.path.join("experiment", file_name),
         os.path.join(results_dir, file_name))
