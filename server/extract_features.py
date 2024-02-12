import numpy as np
import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import os

# Load the VGG16 model
model = vgg16.VGG16()

# remove the prediction layer
model = Model(model.inputs, model.layers[-2].output)


def extract(directory = None, images = None):
    '''
    `directory` must end with /

    `images` - np.array of images
    '''
    features = []

    if directory:
        for image_file in os.listdir(directory):
            image_path = os.path.join(directory, image_file)
            img = img_to_array(load_img(image_path, target_size=(224, 224)))

            # reshape the image for the model
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            # preprocess the image
            pp = preprocess_input(img)
            # extract features
            features.append(model.predict(pp, verbose=0))

    elif not images is None:
        for img in images:
            img = img_to_array(img.resize((224, 224)))

            # reshape the image for the model
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            # preprocess the image
            pp = preprocess_input(img)
            # extract features
            features.append(model.predict(pp, verbose=0))

    return np.array(features).reshape(-1, 4096)
