from scripts.utility import fid_score
import os
import argparse

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import swirl
from skimage.filters import gaussian
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import lpips


# save plots for latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


class FIDTestFolder(Dataset):
    def __init__(self, root, img_size=(32, 64), transform=None, deformation=None, deformation_kwargs=None):
        super(FIDTestFolder, self).__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(self.img_size),
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

        # deformations
        if self.deformation and self.deformation_kwargs:
            if self.deformation == "gaus_noise":
                alpha = self.deformation_kwargs
                noise = np.random.normal(
                    size=img_deformed.shape, loc=0, scale=1)
                noise = (noise - noise.min()) / (noise.max() - noise.min())
                img_deformed = (1 - alpha)*img_deformed + alpha*noise
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
                    img_deformed, radius=48, strength=strength)
            elif self.deformation == "blur":
                sigma = self.deformation_kwargs
                img_deformed = gaussian(
                    img_deformed, sigma=sigma, multichannel=True)

            # make sure it is between [0, 1]
            img_deformed = np.clip(img_deformed, 0, 1).astype(np.float32)

        if self.transform:
            img = self.transform(img)
            img_deformed = self.transform(img_deformed)

        return img, img_deformed


# test dataset
parser = argparse.ArgumentParser("Testing Script for FID and LPIPS")
parser.add_argument(
    "--potsdam_dir", default="../potsdam_data/potsdam_cars_corrected", help="path to potsdam cars")
parser.add_argument("--model", default="vgg", choices=["vgg", "alex"],
                    help="lpips back bone model, vgg or alexnet")
parser.add_argument("--experiment", default="fid", choices=["fid", "lpips"])
parser.add_argument("save_dir")
args = parser.parse_args()

potsdam_dir = args.potsdam_dir
save_dir = args.save_dir
experiment = args.experiment
loss = lpips.LPIPS(net=args.model).cuda()
os.makedirs(save_dir,  exist_ok=True)

# Deformations and their parameters
deformations = ["swirl", "blur", "salt_pepper_noise", "gaus_noise"]
params = [[0, 3, 5, 7, 10],
          [0, 1, 3, 5, 7],
          [0, 0.01, 0.05, 0.1, 0.25],
          [0, 0.1, 0.25, 0.50, 0.75]]

# save scores to dict so we can plot and save them
scores = dict((deformation, []) for deformation in deformations)


for param, deformation in zip(params, deformations):

    fig, axis = plt.subplots()
    fig.set_size_inches(8, 5)

    images = []
    scores = []

    for p in param:
        dataset = FIDTestFolder(
            potsdam_dir, deformation=deformation, deformation_kwargs=p)

        # get deformed image
        _, img_deformed = dataset[0]
        img_deformed = (img_deformed.numpy().transpose(1, 2, 0) + 1) / 2
        images.append(img_deformed)

        dataloader = DataLoader(dataset, 1024, shuffle=True, num_workers=4)
        imgs1, imgs2 = next(iter(dataloader))

        if experiment == "fid":
            # compute fid score
            score = fid_score(imgs1, imgs2, device="cuda:0",
                              layer_idx=33, use_bn=True)
            print(f"Deformation {deformation} - Param {p} - FID Score {score}")

        elif experiment == "lpips":
            # compute lpips
            score = loss(imgs1.cuda(), imgs2.cuda(),
                         normalize=True).mean().item()
            print(
                f"Deformation {deformation} - Param {p} - LPIPS Score {score}")

        scores.append(score)

    y_max = max(scores)
    # add exemplary image
    for i, img in enumerate(images):
        imagebox = OffsetImage(img, zoom=1.5)
        ab = AnnotationBbox(
            imagebox, (i, y_max + (3 if experiment == "fid" else 0.25)))
        axis.add_artist(ab)

    axis.plot(range(len(scores)), scores, marker="*")
    plt.draw()
    axis.set_xlim(-0.5, len(scores) - 0.5)
    axis.set_ylim(0, y_max + (4 if experiment == "fid" else 0.3))
    axis.set_ylabel("FID Score" if experiment ==
                    "fid" else "LPIPS Score", fontsize=26)
    axis.set_xlabel("Disturbance Level", fontsize=26)
    axis.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment}_{deformation}.pgf"))
    plt.savefig(os.path.join(save_dir, f"{experiment}_{deformation}.png"))
    plt.clf()
    plt.close()
