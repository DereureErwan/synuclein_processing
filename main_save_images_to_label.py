import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances  
import torch
from tqdm import tqdm

import os
import sys

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import json
import os

from solo.args.umap import parse_args_umap
from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS


sys.argv = ['main_umap', '--dataset', 'custom', '--train_data_path', 'path/to/train/data', '--val_data_path', 'path/to/val/data',
            '--batch_size', '32', '--num_workers', '4', '--pretrained_checkpoint_dir', 'path/to/pretrained/checkpoint', '--no_labels']


args = parse_args_umap()



def get_image_by_index(data_loader, target_index):
    current_index = 0
    for inputs, *_ in data_loader:
        batch_size = inputs.size(0)
        if current_index + batch_size > target_index:
            relative_idx = target_index - current_index
            return inputs[relative_idx]
        current_index += batch_size
    raise IndexError("Index out of range.")

def show_similar_images_from_loader(features_np, query_index, n_neighbors=5, metric='cosine'):

    # Step 1: Get query embedding
    query_feat = features_np[query_index].reshape(1, -1)

    # Step 2: Compute distances
    if metric == 'cosine':
        dists = cosine_distances(query_feat, features_np)[0]
    else:
        dists = euclidean_distances(query_feat, features_np)[0]

    nearest_indices = np.argsort(dists)[1:n_neighbors+1]


    return query_index, nearest_indices

# build paths
ckpt_dir = Path(args.pretrained_checkpoint_dir)
args_path = ckpt_dir / "args.json"
ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

# load arguments
with open(args_path) as f:
    method_args = json.load(f)
cfg = OmegaConf.create(method_args)

# build the model
model = (
    METHODS[method_args["method"]]
    .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
    .backbone
)

print(args.train_data_path)
print(args.val_data_path)
# prepare data
train_loader, val_loader = prepare_data(
    args.dataset,
    train_data_path=args.train_data_path,
    val_data_path=args.val_data_path,
    data_format=args.data_format,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    auto_augment=False,
    no_labels=args.no_labels,
)


device = "cuda:0"
model = model.to(device)

Y = []
data = []

        # set module to eval model and collect all feature representations
model.eval()
with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Collecting features"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feats = model(x)
        data.append(feats.cpu())
        Y.append(y.cpu())
model.train()

data = torch.cat(data, dim=0).numpy()
Y = torch.cat(Y, dim=0)
num_classes = len(torch.unique(Y))
Y = Y.numpy()



closest_samples_indexs = [] # Manually identified indices of most distinct samples per class (from KMeans centroids or other method)


closest_samples_indexs = np.array(closest_samples_indexs).flatten()

images_to_label = closest_samples_indexs.copy()

print("Looking for similar images")
for cluster_centroid in closest_samples_indexs:
    print(cluster_centroid)
    query, similar_images = show_similar_images_from_loader(data, cluster_centroid, 125)
    images_to_label = np.append(images_to_label, np.array(similar_images))


np.save("images_to_label.npy", images_to_label)