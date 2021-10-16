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
parammeters = [({"n_layers": 5, "init_channels": 1024, "bn_mode": "default"},
                {"base_channels": 32, "n_layers": 5, "heat_map": True, "heat_map_layer": 5, "bn_mode": "default"}),
               ({"n_layers": 5, "init_channels": 1024, "bn_mode": "default"},
                {"base_channels": 32, "n_layers": 5, "heat_map": True, "heat_map_layer": 4, "bn_mode": "default"})]


img_dim = (3, 64, 128)
batch_size = 64
max_epochs = 1000
data_dir = "/scratch/s7hialtu/potsdam_cars_all"
results_dir = "/scratch/s7hialtu/big_image"
interval = 25

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars_all"
    results_dir = "logs"

# data aug
contrast = [1, 1.25]
hue = [-0.1, 0.1]

transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(
                                    hue=hue, contrast=contrast),
                                transforms.Normalize([0.5], [0.5])])

for generator_param, discriminator_param in parammeters:
    model = GAN(img_dim,
                discriminator_params=discriminator_param,
                generator_params=generator_param)

    potsdam = PostdamCarsDataModule(
        data_dir, img_size=img_dim[1:], batch_size=batch_size, transform=transform)
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
