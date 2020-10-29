import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
labels = ts_spikes[1:len(ts_spikes),:]
labels=np.vstack([labels, ts_spikes[len(ts_spikes)-1,:]])


def PCA_(data,labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    plt.figure(figsize=(16,10))
    plt.scatter(pca_result[:,0],pca_result[:,1])
    plt.savefig("PCA.png")

def TSNE_(data):
    tsne = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=(16,10))
    plt.scatter(tsne[:,0],label="TSNE 2-d one")
    plt.scatter(tsne[:,1],label="TSNE 2-d two")
    plt.savefig("TSNE.png")

def main():
    PCA_(ts_spikes,labels)
    # TSNE_(ts_spikes)
if __name__ == "__main__":
    main()