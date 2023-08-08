import umap
from abc import ABC, abstractmethod

class Reducer:

    def __init__(self):
        pass

    def reduce_features(self, features):
        reducer = umap.UMAP(random_state=42)
        reducer.fit(features)
        embedding = reducer.transform(features)
        return embedding[:, 0], embedding[:, 1]
