import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')

def PCA(data):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    print(components.size)

def TSNE(data):
    tsne = TSNE(n_components=2).fit_transform(data)
    print(tsne.size)

def main():
    PCA(ts_spikes)
    TSNE(ts_spikes)
if __name__ == "__main__":
    main()