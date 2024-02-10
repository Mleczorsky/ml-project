import tensorflow.keras.applications.vgg16 as vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import os


# Load the VGG16 model
model = vgg16.VGG16()

# remove the prediction layer
model = Model(model.inputs, model.layers[-2].output)


def extract(directory = 'images/'):
    features = {}

    for image_file in os.listdir(directory):
        image_path = os.path.join(directory, image_file)
        img = img_to_array(load_img(image_path, target_size=(224, 224)))

        # reshape the image for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # preprocess the image
        pp = preprocess_input(img)
        # extract features 
        features[image_path] = model.predict(pp, verbose=0)

    return features