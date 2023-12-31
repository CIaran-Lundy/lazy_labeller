# for loading/processing the images
from keras.preprocessing.image import load_img
from data_loaders import LoadBatchFromDirectory
# models


# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else

import numpy as np
import plotly.express as px
from random import randint
import pandas as pd

import umap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class Clusterer:

    def __init__(self, x):
        self.x = x
        self.k = self.find_k(x)
        self.k_means = self.fit()
        self.centroids = self.k_means.cluster_centers_
        self.labels = self.k_means.labels_

    def fit(self):
        # cluster feature vectors
        kmeans = KMeans(n_clusters=self.k, random_state=22)
        kmeans.fit(self.x)
        return kmeans

    @staticmethod
    def get_clusters(self):
        # holds the cluster id and the images { id: [images] }
        groups = {}
        for file, cluster in zip(filenames, kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)

    @staticmethod
    def find_k(x):
        sse = []
        list_k = list(range(3, 50))

        for k in list_k:
            km = KMeans(n_clusters=k, random_state=22)
            km.fit(x)

            sse.append(km.inertia_)

        # df = pd.DataFrame(dict(k=list_k, sum_squared_error=sse))
        # fig = px.scatter(df, x='k', y='sum_squared_error')
        # fig.show()

        kl = KneeLocator(list_k, sse, curve='convex', direction='decreasing')

        return kl.elbow


#if __name__ == '__main__':
#    batch_size = 10
#
#    batches = LoadBatchFromDirectory(directory='/', batch_size=batch_size)
#
#    for batch in batches.load_batches():
#
#    fe = FeatureExtractor(batch, (100, 100, 3))
#    features = fe.get_features()
#
#    reducer = umap.UMAP(random_state=42)
#    reducer.fit(features)
#
#    embedding = reducer.transform(features)
#
#    clust = Clusterer(embedding)
#
#    cluster_labels = clust.labels
#
#    df = pd.DataFrame(dict(x=embedding[:, 0], y=embedding[:, 1], labels=labels, cluster=cluster_labels, paths=paths))
#
#    #fig = px.scatter(df, x='x', y='y', symbol='labels', color='cluster', hover_data=['labels', 'paths'])
#    #fig.show()
#
#    clusters_df = pd.DataFrame()
#    dfs = df.groupby('cluster')
#    for cluster, d in dfs:
#        d['var'] = np.var(d['x']) / np.var(d['y'])
#        clusters_df = pd.concat([clusters_df, d])
#
#    grouped = clusters_df.groupby('var')


    #for i, gdf in grouped:
    #    print(i)
    #    paths = list(gdf['paths'])
    #    print(paths)
    #    for path in paths


