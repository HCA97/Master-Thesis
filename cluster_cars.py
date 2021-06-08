import argparse
import os

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from scripts import *

parser = argparse.ArgumentParser("Cluster cars using VGG-16 features")
parser.add_argument(
    "--potsdam_dir", default="../potsdam_data/potsdam_cars", help="potsdam cars path")
parser.add_argument(
    "--save_dir", default="experiments/cluster_cars", help="save path")
args = parser.parse_args()

potsdam_dir = args.potsdam_dir
save_dir = args.save_dir
numpy_path = os.path.join(save_dir, "visualization.npy")

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(numpy_path):

    potsdam_dataset = PostdamCarsDataModule(potsdam_dir, batch_size=1024)
    potsdam_dataset.setup()
    dataloader = potsdam_dataset.train_dataloader()

    feat = []
    cars = []
    for imgs, _ in dataloader:

        act = vgg16_get_activation_maps(
            imgs, layer_idx=33, device="cuda:0", normalize_range=(-1, 1), use_bn=True)

        cars.append(imgs.detach().cpu().numpy().transpose(0, 2, 3, 1))
        feat.append(np.squeeze(act.detach().cpu().numpy()))

    feat = np.concatenate(feat, axis=0)  # feats
    cars = (np.concatenate(cars, axis=0) + 1) / 2  # cars
    feat_2 = TSNE(n_components=2).fit_transform(feat)  # tsne

    save_dict = {"tsne": feat_2, "feat": feat, "cars": cars}
    np.save(numpy_path, save_dict)

else:
    save_dict = np.load(numpy_path, allow_pickle=True).item()
    feat_2, feat, cars = save_dict["tsne"], save_dict["feat"], save_dict["cars"]

# find optimal n_clusters
min_clusters = 1
max_clusters = 15
sse = []
for i in range(min_clusters, max_clusters):
    clf = KMeans(i).fit(feat_2)
    sse.append(clf.inertia_)

plt.figure()
plt.plot(range(min_clusters, max_clusters), sse, "-*")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.xticks(range(min_clusters, max_clusters))
plt.title("Kmeans Cluster Search")
plt.savefig(os.path.join(save_dir, "sse_results.png"))
plt.clf()
plt.close()

# n_clusters 6 is good
n_clusters = 6
clf = KMeans(n_clusters).fit(feat_2)
labels = clf.labels_
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
for i in range(n_clusters):
    plt.plot(feat_2[labels == i, 0],
             feat_2[labels == i, 1], ".", color=colors[i])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("TSNE Visualization")
plt.savefig(os.path.join(save_dir, "tsne.png"))

# plot cars 10 per each one
plt.figure(figsize=(15, 15))
n_cars = 20

for i in range(n_clusters):
    imgs = cars[labels == i]
    n, h, w, c = imgs.shape
    idx = np.random.choice(n, n_cars, replace=False)
    img = imgs[idx, :, :, :].reshape(n_cars*h, w, c)
    plt.subplot(1, n_clusters, i+1)
    plt.imshow(img)
    plt.title(f"Cluster-{i+1}", color=colors[i], fontsize=24)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cluster_cars.png"))
