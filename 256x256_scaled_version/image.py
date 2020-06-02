from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


def read_image(file, bounds):
    image = open_image(file, bounds)
    image = normalize_image(image)
    return image


def open_image(file, bounds):
    image = Image.open(file)
    image = image.crop(bounds)
    image = image.resize((64, 64))
    return np.array(image)


# Normalization, [-1,1] Range
def normalize_image(image):
    image = np.asarray(image, np.float32)
    image = image / 127.5 - 1
    return img_to_array(image)


# Restore 0..255 Range
def denormalize_image(image):
    return ((image+1)*127.5).astype(np.uint8)