import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch as th
from skimage.transform import swirl
from skimage.filters import gaussian

from utility import fid_score


class FIDTestFolder(Dataset):
    def __init__(self, root, img_size=(32, 64), transform=None, deformation=None, deformation_kwargs=None):
        super(FIDTestFolder, self).__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(self.img_size),
                                             transforms.Normalize([0.5], [0.5])])
        if transform:
            self.transform = transform
        self.deformation = deformation
        self.deformation_kwargs = deformation_kwargs

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        img_deformed = np.array(img.copy(), dtype=np.float32) / 255.0

        # rotate
        if self.deformation and self.deformation_kwargs:
            if self.deformation == "gaus_noise":
                mean, std = 0, self.deformation_kwargs
                img_deformed += np.random.normal(
                    size=img_deformed.shape, loc=mean, scale=std)
            elif self.deformation == "salt_pepper_noise":
                ratio = np.clip(self.deformation_kwargs, 0, 1)
                r, c = img_deformed.shape[:2]
                idx = np.random.choice(r*c, size=int(r*c*ratio), replace=False)
                rows, cols = np.unravel_index(idx, (r, c))
                img_deformed[rows, cols, :] = np.random.randint(
                    0, 2, size=(len(idx), 1))
            elif self.deformation == "swirl":
                strength = self.deformation_kwargs
                img_deformed = swirl(
                    img_deformed, radius=32, strength=strength)
            elif self.deformation == "blur":
                sigma = self.deformation_kwargs
                img_deformed = gaussian(
                    img_deformed, sigma=sigma, multichannel=True)

            # make sure it is between [0, 1]
            img_deformed = np.clip(img_deformed, 0, 1)

        if self.transform:
            img = self.transform(img)
            img_deformed = self.transform(img_deformed)

        return img, img_deformed


# test dataset
potsdam_dir = "../potsdam_data/potsdam_cars"

deformations = ["swirl", "blur", "salt_pepper_noise", "gaus_noise"]
params = [[3, 5, 7, 10], [0.1, 1, 3, 5],
          [0.01, 0.05, 0.1, 0.25], [0.01, 0.05, 0.1, 0.25]]
fid_scores = dict((deformation, []) for deformation in deformations)

fig, axes = plt.subplots(nrows=len(params[0]), ncols=len(deformations) + 1)
fig.set_size_inches(10, 6)
for i, (param, deformation) in enumerate(zip(params, deformations)):
    for j, p in enumerate(param):
        dataset = FIDTestFolder(potsdam_dir, deformation=deformation,
                                deformation_kwargs=p)
        img, img_deformed = dataset[0]

        # compute fid score
        dataloader = DataLoader(dataset, 1024, shuffle=True, num_workers=4)
        imgs1, imgs2 = next(iter(dataloader))
        fid = fid_score(imgs1, imgs2, device="cuda:0")
        print(
            f"Deformation {deformation} - Param {p} - FID Score {fid}")
        fid_scores[deformation].append(fid)

        # convert to numpy
        img_deformed = (img_deformed.detach().cpu(
        ).numpy().transpose(1, 2, 0) + 1) / 2
        img = (img.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2

        # plot results
        axes[j][i].imshow(img_deformed)
        axes[j][i].set_title(f"{deformation}_{p}")
        axes[j][4].imshow(img)
        axes[j][4].set_title("Original")

# set axis parameters
for row in axes:
    for col in row:
        col.get_xaxis().set_visible(False)
        col.get_yaxis().set_visible(False)
plt.tight_layout()

# plot fid
plt.figure()
for i, deformation in enumerate(fid_scores):
    plt.subplot(2, 2, i+1)
    plt.title(f"FID Score {deformation}")
    y = [0] + fid_scores[deformation]
    x = np.arange(0, len(y), 1)
    plt.plot(x, y, marker="*")
    plt.xlabel("Disturbance")
    plt.ylabel("FID Score")
plt.tight_layout()

plt.show()
