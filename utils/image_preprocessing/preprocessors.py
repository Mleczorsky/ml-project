import numpy as np
from PIL import Image


def grayscale_Image(image: Image, shape=(300, 200)) -> np.array:
    return np.array(image.resize(shape).convert('L')).flatten()


def grayscale_Dataset(dataset: np.array, image_shape=(300, 200)) -> np.array:
    """
        X: `np.array[PIL.Image]`
    """
    return np.array(list(map(lambda img: grayscale_Image(img, image_shape), dataset)))


def normalize_dataset(X : np.array) -> np.array:
    """
    Normalizes dataset given as `np.array[ np.array ]` inner ones being pictures.
    """
    return X / X.mean(axis=1).reshape(-1, 1)


def preprocess_dataset(X: np.array, image_shape=(300, 200)) -> np.array:
    """
        X: `np.array[PIL.Image]`
    """
    return normalize_dataset(grayscale_Dataset(X, image_shape))
