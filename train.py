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
generator_params = {"n_layers": 5, "init_channels": 512, "inject_noise": True, "use_spectral_norm": True,
                    "bn_mode": "default", "act": "leakyrelu", "last_layer_kernel_size": 3}
discriminator_params = {"base_channels": 32, "use_spectral_norm": True,
                        "n_layers": 5, "bn_mode": "default"}

img_dim = (3, 64, 128)
batch_size = 64
max_epochs = 1000
interval = 25
alpha = 0.1
use_gp = True

data_dir = "/scratch/s7hialtu/potsdam_cars_all"
results_dir = "/scratch/s7hialtu/dcgan_stylegan"

if not os.path.isdir(data_dir):
    data_dir = "../potsdam_data/potsdam_cars_all"
    results_dir = "logs"

# DATA AUG FOR
transform = transforms.Compose([transforms.Resize(img_dim[1:]),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(
                                    hue=[-0.1, 0.1], contrast=[1, 1.25]),
                                transforms.Normalize([0.5], [0.5])])

model = GAN(img_dim, discriminator_params=discriminator_params, fid_interval=interval, disc_model="basic",
            generator_params=generator_params, gen_model="stylegan", alpha=alpha, use_gp=use_gp)

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
copyfile(file_name,
         os.path.join(results_dir, file_name))
