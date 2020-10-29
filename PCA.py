import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

wget -O donkeykong.5000.ts.spikes.npy.gz https://s3-us-west-2.amazonaws.com/ericmjonas-public/data/neuroproc/donkeykong.5000.ts.spikes.npy.gz
gunzip -f donkeykong.5000.ts.spikes.npy.gz
ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
pca = PCA(n_components=2)
pca.fit(ts_spikes)

