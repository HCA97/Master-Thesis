import os
import argparse

import tqdm
import torch as th
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from scripts import *

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir", help="experiment folder")
parser.add_argument("--interval", default=25, type=int,
                    help="interval between two epochs")
args = parser.parse_args()

max_epochs = 1000
checkpoints = [
    i-1 for i in range(args.interval, max_epochs, args.interval)] + [max_epochs-1]
checkpoint_path = os.path.join(args.checkpoint_dir, "epoch={}.ckpt")

ppls = []
for checkpoint in checkpoints:
    path = checkpoint_path.format(checkpoint)
    if os.path.exists(path):
        model = GAN.load_from_checkpoint(path)
        model.eval()
        model.cuda()

        generator = model.generator_avg if model.hparams.moving_average else model.generator

        ppl = perceptual_path_length(
            generator, n_samples=1024*5, net="vgg", device=model.device, batch_size=128)
        print(f"Epoch {checkpoint} - PPL {ppl}")

        ppls.append(ppl)

# plt.figure()
plt.plot(checkpoints[:len(ppls)], ppls, marker="*")
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("PPL Score", fontsize=18)
plt.title("PPL Score", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
