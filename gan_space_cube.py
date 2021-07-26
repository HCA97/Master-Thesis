import os
import pickle
import argparse

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from scripts import *


@th.no_grad()
def get_inner_act(model, n_samples=10000, layer="linear"):
    z = th.normal(0, 1, size=(
        n_samples, model.generator.latent_dim), device=model.device)
    l = model.generator.l1(z)
    if layer == "conv-1":
        l = l.view(l.shape[0],  model.generator.init_channels,
                   model.generator.init_height,  model.generator.init_width)
        l = model.generator.conv_blocks[0](l)
        l = model.generator.conv_blocks[1](l)
        l = model.generator.conv_blocks[2](l)
        l = l.view(l.shape[0], -1)

    X = np.squeeze(l.detach().cpu().numpy())
    y = np.squeeze(z.detach().cpu().numpy())
    return X, y


# input
parser = argparse.ArgumentParser("GAN Space Cube")
parser.add_argument("model_checkpoint")
parser.add_argument("save_path")
parser.add_argument("--n_samples", type=int, default=5000)
parser.add_argument("--layer", choices=["linear", "conv-1"], default="linear")
parser.add_argument("--n_components",  type=int, default=4096)
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
save_path = args.save_path
n_samples = args.n_samples
n_time = 5
layer = args.layer
n_components = args.n_components

model = GAN.load_from_checkpoint(model_checkpoint)
model.eval()
# model.cuda()


pca_path = os.path.join(save_path, layer, "pca.pkl")

if not os.path.exists(pca_path):
    pca = IncrementalPCA(n_components=n_components)
    os.makedirs(os.path.join(save_path, layer), exist_ok=True)

    for i in range(n_time):
        print("Iteration - ", i+1)
        X, y = get_inner_act(model, n_samples, layer=layer)

        # save the data
        np.save(os.path.join(save_path, layer, f"y{i+1}.npy"), y)
        np.save(os.path.join(save_path, layer, f"X{i+1}.npy"), X)

        pca.partial_fit(X)

    # save the pca
    f = open(pca_path, "wb")
    pickle.dump(pca, f)
    f.close()
else:
    f = open(pca_path, 'rb')
    pca = pickle.load(f)
    f.close()
