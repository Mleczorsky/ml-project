import numpy as np
from PIL import Image


def grayscale_Image(image: Image) -> np.array:
    return np.array(image.resize((300, 200)).convert('L')).flatten()


def grayscale_Dataset(dataset: np.array[Image]) -> np.array:
    return np.array(list(map(grayscale_Image, dataset)))


def average(X: np.array[Image]) -> np.array:
    X = grayscale_Dataset(X)
    return X / X.mean(axis=1).reshape(-1, 1)
