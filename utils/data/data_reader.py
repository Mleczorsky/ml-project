import os
import numpy as np
from PIL import Image
from .data_augmentation import augmentation_transform

CLASS_NAMES = ['rock', 'paper', 'scissors']

def read_dataset(dirpath, n_aug=0) -> np.array:
    """
    Return dataset as `np.array[PIL.Image]`
    """
    X, y = [], []

    for class_id, class_name in enumerate(CLASS_NAMES):
        for filename in os.listdir(f'{dirpath}/{class_name}'):
            with Image.open(f'{dirpath}/{class_name}/{filename}') as original_image:
                aug_images = [original_image.copy()] + [augmentation_transform(original_image) for _ in range(n_aug)]

                for image in aug_images:
                    X.append(image)
                    y.append(class_id)

    permutation = np.random.permutation(len(y))
    return np.array(X, dtype=object)[permutation], np.array(y)[permutation]
