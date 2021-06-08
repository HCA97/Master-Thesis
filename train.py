import os

from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from scripts.models import *
from scripts.utility import *
from scripts.dataloader import *
from scripts.callbacks import *

# POTSDAM CARS
generator_params = {"n_layers": 4, "init_channels": 512}
discriminator_params = {"base_channels": 32, "n_layers": 4, "heat_map": True}

img_dim = (3, 32, 64)
batch_size = 64
max_epochs = 500

# GP
use_gp = False
alpha = 0.1

# Data Aug
contrast = [1, 1.25]
hue = [-0.1, 0.1]
transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(
                                    hue=hue, contrast=contrast),
                                transforms.Normalize([0.5], [0.5])])

data_dir = "/scratch/s7hialtu/potsdam_cars"
results_dir = "/scratch/s7hialtu/dcgan_heatmap"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars"
    results_dir = "logs"

model = GAN(img_dim, discriminator_params=discriminator_params,
            generator_params=generator_params, use_gp=use_gp, alpha=alpha)

potsdam = PostdamCarsDataModule(
    data_dir, img_size=img_dim[1:], batch_size=batch_size, transform=transform)
potsdam.setup()

callbacks = [
    TensorboardGeneratorSampler(
        epoch_interval=25, num_samples=batch_size, normalize=True),
    LatentDimInterpolator(interpolate_epoch_interval=25, num_samples=10),
    ModelCheckpoint(period=25, save_top_k=-1, filename="{epoch}")
]

# Apparently Trainer has logger by default
trainer = pl.Trainer(default_root_dir=results_dir, gpus=1, max_epochs=max_epochs,
                     callbacks=callbacks, progress_bar_refresh_rate=20)
trainer.fit(model, datamodule=potsdam)
