import numpy as np
from collections import Counter


def gym_preprocess(image):
    image = image[34:194, :, :]  # 160, 160, 3
    image = np.mean(image, axis=2, keepdims=False)  # 160, 160
    image = image[::2, ::2]  # 80, 80
    image = image / 256
    constant = Counter(image).most_common()[0][0]  # constant, eg: 90
    image = image - constant / 256  # 80, 80
    return image
