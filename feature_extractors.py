from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np


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
