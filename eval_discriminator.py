import argparse
import os

import torch as th
import matplotlib.pyplot as plt

from scripts import *

parser = argparse.ArgumentParser(
    "Evaluting Discriminator call this after ``eval_dcgan.py`` or ``eval_pix2pix.py``")
parser.add_argument(
    "--train_dir", default="../potsdam_data/potsdam_cars", help="training cars path")
parser.add_argument(
    "--test_dir", default="../potsdam_data/potsdam_cars_test", help="test cars path")
parser.add_argument("--n_samples", type=int, default=125,
                    help="number of samples")
parser.add_argument("results_dir", help="where to save")
parser.add_argument("model_path", help="gan path")
parser.add_argument("fake_dir", help="fake images")
args = parser.parse_args()

results_dir = args.results_dir
test_dir = args.test_dir
train_dir = args.train_dir
fake_dir = args.fake_dir
n_samples = args.n_samples
model_path = args.model_path

os.makedirs(results_dir, exist_ok=True)

# load model
model = GAN.load_from_checkpoint(model_path)
model.eval()

img_dim = model.hparams.img_dim[1:]

# load train dataset
train = PostdamCarsDataModule(
    train_dir, img_size=img_dim, batch_size=n_samples)
train.setup()
train_loader = train.train_dataloader()
train_cars = next(iter(train_loader))[0]

# load test dataset
test = PostdamCarsDataModule(
    test_dir, img_size=img_dim, batch_size=n_samples)
test.setup()
test_loader = test.train_dataloader()
test_cars = next(iter(test_loader))[0]

# load fake dataset
fake = PostdamCarsDataModule(
    fake_dir, img_size=img_dim, batch_size=n_samples)
fake.setup()
fake_loader = fake.train_dataloader()
fake_cars = next(iter(fake_loader))[0]

# compute confidences for each image
with th.no_grad():
    train_cars_pred = model.discriminator(train_cars).detach().cpu().numpy()
    test_cars_pred = model.discriminator(test_cars).detach().cpu().numpy()
    fake_cars_pred = model.discriminator(fake_cars).detach().cpu().numpy()

# plot box plot for each of them
plt.boxplot(
    [train_cars_pred[:, 0], test_cars_pred[:, 0], fake_cars_pred[:, 0]])
plt.xticks([1, 2, 3], ["Train Cars", "Test Cars", "Fake Cars"])
plt.yticks([i / 10 for i in range(10+1)])
plt.title("Discriminator's Prediction")
# plt.ylabel("Discriminator's Prediction")
plt.savefig(os.path.join(results_dir, "eval_discriminator.png"))
