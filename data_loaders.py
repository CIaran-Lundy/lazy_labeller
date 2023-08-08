from keras.preprocessing.image import img_to_array
import os
from PIL import Image
from abc import ABC, abstractmethod


class Loader(ABC):
    """
    abstract base class for loaders
    """
    @abstractmethod
    @staticmethod
    def load(directory):
        pass


class LoadFromDirectory:

    def __init__(self, directory):
        self.directory = directory
        self.data = []
        self.labels = []
        self.image_paths = []

    @staticmethod
    def load(directory):
        data = []
        paths = []

        for path in os.listdir(directory):
            img = img_to_array(Image.open(f'{directory}/{path}'))
            if img.shape == (100, 100, 3):
                data.append(img)
                paths.append(path)
        return data, paths
