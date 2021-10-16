import argparse
import os

import tqdm
from torchvision.utils import make_grid, save_image
import cv2

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from scripts import *

# save plots for latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

parser = argparse.ArgumentParser("Cluster cars using VGG-16 features")
parser.add_argument(
    "--potsdam_dir", default="../potsdam_data/potsdam_cars_all", help="potsdam cars path")
parser.add_argument(
    "--save_dir", default="experiments/cluster_cars", help="save path")
parser.add_argument("--n_cars", default=30,
                    help="number of cars per cluster to plot")
parser.add_argument("--resample", action="store_true", dest="resample",
                    help="resample dataset according to cluster size or not.")
parser.add_argument("--use_pca", action="store_true")
parser.set_defaults(resample=False)
parser.set_defaults(use_pca=False)
args = parser.parse_args()

potsdam_dir = args.potsdam_dir
save_dir = args.save_dir
n_cars = args.n_cars
resample = args.resample
use_pca = args.use_pca

numpy_path = os.path.join(save_dir, "visualization.npy")

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(numpy_path):

    potsdam_dataset = PostdamCarsDataModule(potsdam_dir, batch_size=1024)
    potsdam_dataset.setup()
    dataloader = potsdam_dataset.train_dataloader()

    feat = []
    cars = []
    for imgs, _ in tqdm.tqdm(dataloader, desc="Featrues"):

        cars.append(imgs.detach().cpu().numpy().transpose(0, 2, 3, 1))

        if not use_pca:
            act = vgg16_get_activation_maps(
                imgs, layer_idx=33, device="cuda:0", normalize_range=(-1, 1), use_bn=True)
            feat.append(np.squeeze(act.detach().cpu().numpy()))

    cars = (np.concatenate(cars, axis=0) + 1) / 2  # cars
    if not use_pca:
        feat = np.concatenate(feat, axis=0)  # feats
    else:
        feat = PCA(n_components=100).fit_transform(cars.reshape(len(cars), -1))
    feat_2 = TSNE(n_components=2).fit_transform(feat)  # tsne

    # davies bouldin score
    clusters = [i for i in range(2, 21)]
    db_score = []
    for cluster in tqdm.tqdm(clusters, desc="Cluster"):
        clf = KMeans(cluster).fit(feat)
        labels = clf.predict(feat)
        db_score.append(davies_bouldin_score(feat, labels))

    # clusters
    n_clusters = clusters[np.argmin(db_score)]
    clf = KMeans(n_clusters).fit(feat)
    labels = clf.predict(feat)

    # save everthing
    save_dict = {"db_score": db_score, "labels": labels,
                 "clusters": clusters, "tsne": feat_2,
                 "feat": feat, "cars": cars}
    np.save(numpy_path, save_dict)

else:
    save_dict = np.load(numpy_path, allow_pickle=True).item()
    feat_2, feat, cars = save_dict["tsne"], save_dict["feat"], save_dict["cars"]
    labels, db_score, clusters = save_dict["labels"], save_dict["db_score"], save_dict["clusters"]

n_clusters = int(np.max(labels))+1

# DB SCORE
plt.figure(figsize=(8, 6))
plt.plot(clusters, db_score, marker="*")
# plt.title("Clustering Score", fontsize=20)
plt.xticks([clusters[i]
            for i in range(0, len(clusters)+1, 2)], fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel("Davies Bouldin Score", fontsize=22)
plt.xlabel("Number of Clusters", fontsize=22)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "db_results.pgf"))
plt.clf()
plt.close()

# TSNE
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.plot(feat_2[labels == i, 0],
             feat_2[labels == i, 1], ".", label=f"Cluster {i+1}")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={'size': 18})
# plt.title("TSN-E Visualization", fontsize=18)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "tsne.pgf"))
plt.clf()
plt.close()


# # CLUSTERS
for i in range(n_clusters):
    imgs = cars[labels == i][:, :, :, [0, 1, 2]]

    idx = np.random.choice(len(imgs), n_cars, replace=False)
    img = th.tensor(imgs[idx].transpose(0, 3, 1, 2))
    grid = make_grid(img[:, :, :, :], nrow=10, padding=2, pad_value=1)
    save_image(grid, os.path.join(save_dir, f"cluster_{i}.png"))


# RESAMPLE
if resample:
    root, name = os.path.split(potsdam_dir)
    resample_save_path = os.path.join(root, name + "_resample")
    os.makedirs(resample_save_path, exist_ok=True)

    idx, size = np.unique(labels, return_counts=True)
    max_size = np.max(size)

    counter = 0
    for i, s in zip(idx, size):
        diff = max_size - s
        cluster = cars[labels == i]

        # save each cluster
        for car in cluster:
            car = (255*car).astype(np.uint8)
            car_path = os.path.join(resample_save_path, "car_%d.png" % counter)
            # cv2.imwrite(car_path, car)
            cv2.imwrite(car_path, cv2.cvtColor(car, cv2.COLOR_RGB2BGR))
            counter += 1

        if diff > 0:
            idx = np.random.choice(len(cluster), diff)
            for car in cluster[idx]:
                car = (255*car).astype(np.uint8)
                car_path = os.path.join(
                    resample_save_path, "car_%d.png" % counter)
                # cv2.imwrite(car_path, car)
                cv2.imwrite(car_path, cv2.cvtColor(car, cv2.COLOR_RGB2BGR))
                counter += 1
