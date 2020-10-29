import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
labels = ts_spikes[1:len(ts_spikes),:]
labels=np.vstack([labels, ts_spikes[len(ts_spikes)-1,:]])
ts_spikes['y'] = labels


def PCA_(data,labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data.drop("y"))
    data['pca-one'] = pca_result[:,0]
    data['pca-two'] = pca_result[:,1] 
    data['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# def TSNE_(data):
#     tsne = TSNE(n_components=2).fit_transform(data)
#     print(tsne.size)

# def main():
#     PCA_(ts_spikes)
#     # TSNE_(ts_spikes)
# if __name__ == "__main__":
#     main()