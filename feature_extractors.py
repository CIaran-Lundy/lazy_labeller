from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np


class FeatureExtractor:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        model = ResNet50(include_top=False, input_shape=input_shape)
        self.model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    def __extract_features(self, image, model):
        reshaped_img = image.reshape(1, *self.input_shape)
        imgx = preprocess_input(reshaped_img)
        features = model.predict(imgx, use_multiprocessing=True)
        return features

    def get_features(self, image_data):
        features = []
        for image in image_data:
            feat = self.__extract_features(image['image_array'], self.model)
            features.append(feat.flatten())
        return np.array(features)

