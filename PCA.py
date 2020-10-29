import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')

def PCA_(data):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
            cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.savefig("PCA.pdf",dpi=150)

def TSNE_(data):
    tsne = TSNE(n_components=2).fit_transform(data)
    print(tsne.size)

def main():
    PCA_(ts_spikes)
    # TSNE_(ts_spikes)
if __name__ == "__main__":
    main()