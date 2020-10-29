import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
pca = PCA(n_components=2)
components = pca.fit_transform(ts_spikes)
print(components.shape())

