# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# models
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
from PIL import Image
import numpy as np
import plotly.express as px
from random import randint
import pandas as pd

import umap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class LoadFromDirectory:

    def __init__(self):
        self.directory = None
        self.data = []
        self.labels = []
        self.image_paths = []

    @staticmethod
    def load(directory):
        data = []
        labels = []
        paths = []

        sub_class = os.listdir(directory)

        for i, sc in enumerate(sub_class):
            class_path = f'{src_path}/{sc}'
            for path in os.listdir(class_path):
                img = img_to_array(Image.open(f'{class_path}/{path}'))
                if img.shape == (100, 100, 3):
                    data.append(img)
                    labels.append(sc)
                    paths.append(path)
        return data, labels, paths


class FeatureExtractor:

    def __init__(self, data, input_shape):
        self.input_shape = input_shape
        self.data = data
        model = ResNet50(include_top=False, input_shape=input_shape)
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    def __extract_features(self, image, model):
        reshaped_img = image.reshape(1, *self.input_shape)
        imgx = preprocess_input(reshaped_img)
        features = model.predict(imgx, use_multiprocessing=True)
        return features

    def get_features(self):
        data = []
        for i, image in enumerate(self.data):
            feat = self.__extract_features(image, self.model)
            data.append(feat.flatten())
        return np.array(data)


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
    # function that lets you view a cluster (based on identifier)
    def view_cluster(cluster):
        plt.figure(figsize=(25, 25))
        # gets the list of filenames for a cluster
        files = groups[cluster]
        # only allow up to 30 images to be shown at a time
        if len(files) > 30:
            print(f"Clipping cluster size from {len(files)} to 30")
            files = files[:29]
        # plot each image in the cluster
        for index, file in enumerate(files):
            plt.subplot(10, 10, index+1)
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')

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


#if __name__ == "__main__":

src_path = "training_data2/"

data, labels, paths = LoadFromDirectory.load(src_path)

fe = FeatureExtractor(data, (100, 100, 3))
features = fe.get_features()

reducer = umap.UMAP(random_state=42)
reducer.fit(features)

embedding = reducer.transform(features)

clust = Clusterer(embedding)

cluster_labels = clust.labels

df = pd.DataFrame(dict(x=embedding[:, 0], y=embedding[:, 1], labels=labels, cluster=cluster_labels, paths=paths))

#fig = px.scatter(df, x='x', y='y', symbol='labels', color='cluster', hover_data=['labels', 'paths'])
#fig.show()


clusters_df = pd.DataFrame()
dfs = df.groupby('cluster')
for cluster, d in dfs:
    d['var'] = np.var(d['x']) / np.var(d['y'])
    clusters_df = pd.concat([clusters_df, d])

grouped = clusters_df.groupby('var')


for i, gdf in grouped:
    print(i)
    paths = list(gdf['paths'])
    print(paths)
    for path in paths


