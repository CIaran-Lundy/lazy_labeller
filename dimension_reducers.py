import umap
from abc import ABC, abstractmethod

class UMAPReducer:

    def __init__(self):
        self.reducer = umap.UMAP(random_state=42)

    def reduce_features(self, features):
        self.reducer.fit(features)
        embedding = self.reducer.transform(features)
        return embedding[:, 0], embedding[:, 1]
