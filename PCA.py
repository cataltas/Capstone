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
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    # ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    # ax.scatter(
    # xs=pca_result[:,0], 
    # ys=pca_result[:,1], 
    # zs=pca_result[:,2], 
    # c=labels, 
    # cmap='tab10'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    plt.figure(figsize=(16,10))
    plt.scatter(pca_result[:,0],label="First Component")
    plt.scatter(pca_result[:,1],label="Second Component")
    plt.scatter(pca_result[:,2],label="Third Component")
    plt.savefig("PCA.png")

def TSNE_(data):
    tsne = TSNE(n_components=2).fit_transform(data)
    print(tsne.size)

def main():
    PCA_(ts_spikes,labels)
    # TSNE_(ts_spikes)
if __name__ == "__main__":
    main()