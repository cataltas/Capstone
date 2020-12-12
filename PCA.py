import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time


x = pd.read_csv("X.csv")
labels = pd.read_csv("y.csv")
x= x.drop("time",axis=1)
labels=labels.drop("time",axis=1)
labels_dict = {}
encoded_labels=[]
c=0
for i,label in labels.iterrows():
    label = str(label)
    if label in labels_dict.keys():
        encoded_labels.append(labels_dict[label])
    else:
        c=c+1
        labels_dict[label]=c
        encoded_labels.append(c)

def PCA_(data,labels):
    N = 20000
    rndperm = np.random.permutation(data.shape[0])
    data_subset = data.iloc[rndperm[:N],:].copy()
    labels_subset = [labels[i] for i in rndperm[:N]]
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(data_subset)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=X_reduced[:,0], y=X_reduced[:,1],
    hue=labels_subset,
    palette=sns.color_palette("hls", len(np.unique(labels_subset))),
    legend="brief",
    alpha=0.3)
    plt.savefig("PCA_labels_2d_2.png")
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

def TSNE_(data,labels):
    N = 20000
    rndperm = np.random.permutation(data.shape[0])
    data_subset = data.iloc[rndperm[:N],:].copy()
    labels_subset = [labels[i] for i in rndperm[:N]] 
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels_subset,
    palette=sns.color_palette("hls", len(np.unique(labels_subset))),
    legend="brief",
    alpha=0.3)
    plt.savefig("TSNE.png")

def main():
    PCA_(x,encoded_labels)
    # TSNE_(x,encoded_labels)
if __name__ == "__main__":
    main()