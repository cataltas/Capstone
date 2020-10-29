import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


ts_spikes = np.load("donkeykong.5000.ts.spikes.npy", mmap_mode='r')
labels = ts_spikes[1:len(ts_spikes),:]
labels=np.vstack([labels, ts_spikes[len(ts_spikes)-1,:]])
labels_dict = {}
encoded_labels=[]
c=0
for i,label in enumerate(labels):
    label = str(label)
    if label in labels_dict.keys():
        encoded_labels.append(labels_dict[label])
    else:
        c=c+1
        labels_dict[label]=c
        encoded_labels.append(c)


def PCA_(data,labels):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels,
            cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.savefig("PCA_labels.png")

# def TSNE_(data):
#     tsne = TSNE(n_components=2).fit_transform(data)
#     plt.figure(figsize=(16,10))
#     plt.scatter(tsne[:,0],tsne[:,1])
#     plt.savefig("TSNE.png")

def main():
    PCA_(ts_spikes,encoded_labels)
    # TSNE_(ts_spikes)
if __name__ == "__main__":
    main()