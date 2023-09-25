import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

width = 40
walk_xlim = (0.2, 0.5)
walk_ylim = (-0.6, 0.6)
jump_xlim = (-5, 5)
jump_ylim = (0.1, 0.6)

def state_coverage(points, dim1_range = (-5, 5), dim2_range = (0.1, 0.6)):
    resolution = 0.01
    num_bins = int(1 / resolution)
    # normalize points
    print(points.shape, points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max())

    points = np.clip((points - np.array([dim1_range[0], dim2_range[0]])) / np.array([dim1_range[1] - dim1_range[0], dim2_range[1] - dim2_range[0]]), 0, 1)
    bins = np.zeros((num_bins, num_bins))
    for point in points:
        bin1 = min(int(point[0]*num_bins), num_bins-1)
        bin2 = min(int(point[1]*num_bins), num_bins-1)
        # print(bin1, bin2, point)
        bins[bin1, bin2] += 1
    score = np.sum(bins > 0) / (num_bins**2)
    return bins, score

parser = argparse.ArgumentParser()
parser.add_argument("--exptid", type=str, default="025-05")
parser.add_argument('--jump', action='store_true', default=False)
parser.add_argument('--tsne', action='store_true', default=False)
parser.add_argument('--histogram', action='store_true', default=False)
parser.add_argument('--traj', action='store_true', default=True)
parser.add_argument('--coverage', action='store_true', default=True)
parser.add_argument('--scatter', action='store_true', default=True)

args = parser.parse_args()
prefix = f"{args.exptid}-" + ("jump" if args.jump else "walk")
print(prefix)
dir_name = f"./data/{prefix}"
os.makedirs(dir_name, exist_ok=True)

exptid = args.exptid
data_raw = np.load(f'./data/{exptid}.npy', allow_pickle=True)
features = data_raw[:, :-1]
labels = data_raw[:, -1]
print(features.shape, labels.shape)

if args.tsne:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    tsne_features = tsne.fit_transform(features)

    fig, ax = plt.subplots()
    unique_labels = np.unique(labels) 
    i = 0
    for label in unique_labels:
        indices = np.where(labels == label)

        x = tsne_features[indices, 0]
        y = tsne_features[indices, 1]

        ax.scatter(x, y, label=label, alpha=0.5, marker="o")
        if i == 12:
            break
        i+= 1
    ax.legend()
    plt.title(f'{exptid} t-SNE Visualization')
    plt.show()

if args.histogram:
    feature_t = features[:, :4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        ax = axes[i//2, i%2]
        f0 = feature_t[labels==i] + 0.5
        uniques, counts = np.unique(f0, axis=0, return_counts=True)
        print(uniques.shape, counts.shape)
        uniques_dec = uniques[:, 0]*8 + uniques[:, 1]*4 + uniques[:, 2]*2 + uniques[:, 3]
        print(uniques_dec.shape)
        hist_values = np.repeat(uniques_dec, counts)

        # Plot the histogram
        ax.hist(hist_values, bins=len(uniques_dec)+1)

        ax.set_xlabel
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Unique Values with Bin Counts")
        
        # Display the plot


if args.traj:
    print("#"*width, "Trajectories", "#"*width)
    def draw_traj(features, labels, xlim, ylim, xlabel, ylabel):
        height = features[:, 0]
        pitch = features[:, 1]
        unique_labels = np.unique(labels) 
        fig, axes = plt.subplots(2, unique_labels.shape[0]//2, figsize=(30, 5))
        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)

            x = height[indices]
            y = pitch[indices]

            ax = axes[i%2, i//2]
            ax.scatter(x, y, label=label, alpha=0.2, marker="o", color="orange")
            
            ax.legend()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(prefix)
        return fig
    if args.jump:
        fig = draw_traj(features, labels, jump_xlim, jump_ylim, "Pitch Vel", "Height")
    else:
        fig = draw_traj(features, labels, walk_xlim, walk_ylim, "Height", "Pitch")

    fig.savefig(f"{dir_name}/{prefix}-traj.png", bbox_inches='tight')

if args.coverage:
    print("#"*width, "State Coverage", "#"*width)
    if args.jump:
        bins, score = state_coverage(features[:, :2], dim1_range=jump_xlim, dim2_range=jump_ylim)
    else:
        bins, score = state_coverage(features[:, :2], dim1_range=walk_xlim, dim2_range=walk_ylim)
    print(f"State Coverage Score: {score}")
    norm_bins = np.clip(bins / np.max(bins), 0, 0.6)
    fig, ax = plt.subplots()
    ax.imshow(norm_bins, cmap="viridis", interpolation="nearest")
    ax.set_title(prefix + f"-{score}")

    fig.savefig(f"{dir_name}/{prefix}-coverage.png", bbox_inches='tight')


if args.scatter:
    def scatter(features, labels, xlim, ylim, xlabel, ylabel):
        height = features[:, 0]
        pitch = features[:, 1]

        fig, ax = plt.subplots()
        unique_labels = np.unique(labels) 
        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)

            x = height[indices]
            y = pitch[indices]

            ax.scatter(x[:], y[:], label=label, alpha=0.2, marker="o")
            
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(prefix)
        return fig
    if args.jump:
        fig = scatter(features, labels, jump_xlim, jump_ylim, "Pitch Vel", "Height")
    else:
        fig = scatter(features, labels, walk_xlim, walk_ylim, "Height", "Pitch")
    fig.savefig(f"{dir_name}/{prefix}-scatter.png", bbox_inches='tight')




