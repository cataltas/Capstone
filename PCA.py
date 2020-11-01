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


# def PCA_(data,labels):
#     pca = PCA(n_components=2)
#     X_reduced = pca.fit_transform(data)
#     plt.figure(figsize=(16,10))
#     sns.scatterplot(
#     x=X_reduced[:,0], y=X_reduced[:,1],
#     hue=labels,
#     palette=sns.color_palette("hls", 8),
#     legend="full",
#     alpha=0.3)
#     plt.savefig("PCA_labels_2d.png")
#     print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

def TSNE_(data,labels):
    tsne = TSNE(n_components=2,n_iter=300)
    tsne_results=tsne.fit_transform(data)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    palette=sns.color_palette("hls", 8),
    legend="full",
    alpha=0.3
    )
    plt.savefig("TSNE.png")

def main():
    # PCA_(ts_spikes,encoded_labels)
    TSNE_(ts_spikes,encoded_labels)
if __name__ == "__main__":
    main()