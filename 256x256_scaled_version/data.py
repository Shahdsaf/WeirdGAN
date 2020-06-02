import shutil
import os

import numpy as np
import xml.etree.ElementTree as et
from tensorflow.keras.preprocessing.image import array_to_img

from image import read_image, denormalize_image
from adversarial_networks_scaled import INPUT_SIZE


def generate_noise(size):
    return np.random.normal(0, 1, size=[size, INPUT_SIZE])


def load_images():
    images = []
    print('Loading Images')
    for breed in os.listdir('data/Annotation/'):
        for dog in os.listdir('data/Annotation/' + breed):
            tree = et.parse('data/Annotation/' + breed + '/' + dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                box = o.find('bndbox')
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)

            bounds = (xmin, ymin, xmax, ymax)
            try:
                image = read_image('data/all-dogs/' + dog + '.jpg', bounds)
                images.append(image)
            except:
                print('No image', dog)

    return np.array(images)


def save_images(generator):
    if not os.path.exists('output'):
        os.mkdir('output')

    noise = generate_noise(10000)
    generated_images = generator.predict(noise)

    for i in range(generated_images.shape[0]):
        image = denormalize_image(generated_images[i])
        image = array_to_img(image)
        image.save( 'output/' + str(i) + '.png')

    shutil.make_archive('images', 'zip', 'output')

