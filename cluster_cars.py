import argparse
import os

import tqdm
from torchvision.utils import make_grid
import cv2

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from scripts import *

parser = argparse.ArgumentParser("Cluster cars using VGG-16 features")
parser.add_argument(
    "--potsdam_dir", default="../potsdam_data/potsdam_cars", help="potsdam cars path")
parser.add_argument(
    "--save_dir", default="experiments/cluster_cars", help="save path")
parser.add_argument("--n_cars", default=30,
                    help="number of cars per cluster to plot")
parser.add_argument("--resample", action="store_true", dest="resample",
                    help="resample dataset according to cluster size or not.")
parser.set_defaults(resample=False)
args = parser.parse_args()

potsdam_dir = args.potsdam_dir
save_dir = args.save_dir
n_cars = args.n_cars
resample = args.resample

numpy_path = os.path.join(save_dir, "visualization.npy")

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(numpy_path):

    potsdam_dataset = PostdamCarsDataModule(potsdam_dir, batch_size=1024)
    potsdam_dataset.setup()
    dataloader = potsdam_dataset.train_dataloader()

    feat = []
    cars = []
    for imgs, _ in tqdm.tqdm(dataloader, desc="Featrues"):

        act = vgg16_get_activation_maps(
            imgs, layer_idx=33, device="cuda:0", normalize_range=(-1, 1), use_bn=True)

        cars.append(imgs.detach().cpu().numpy().transpose(0, 2, 3, 1))
        feat.append(np.squeeze(act.detach().cpu().numpy()))

    feat = np.concatenate(feat, axis=0)  # feats
    cars = (np.concatenate(cars, axis=0) + 1) / 2  # cars
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

# DB SCORE
plt.figure(figsize=(7, 4))
plt.plot(clusters, db_score, marker="*")
plt.title("Clustering Score", fontsize=20)
plt.xticks([clusters[i]
            for i in range(0, len(clusters)+1, 2)], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Davies Bouldin Score", fontsize=18)
plt.xlabel("# of Clusters", fontsize=18)
plt.savefig(os.path.join(save_dir, "db_results.png"))
plt.clf()
plt.close()

# CLUSTERS
n_clusters = int(np.max(labels))+1
for i in range(n_clusters):
    imgs = cars[labels == i]

    idx = np.random.choice(len(imgs), n_cars, replace=False)
    img = th.tensor(imgs[idx].transpose(0, 3, 1, 2))
    grid = make_grid(img, nrow=5, padding=2)
    img = grid.numpy().transpose(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"cluster_{i}.png"))
    plt.clf()
    plt.close()

# TSNE
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.plot(feat_2[labels == i, 0],
             feat_2[labels == i, 1], ".")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("TSNE Visualization", fontsize=18)
plt.savefig(os.path.join(save_dir, "tsne.png"))
plt.clf()
plt.close()

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
