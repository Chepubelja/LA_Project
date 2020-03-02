import numpy as np
from PIL import Image

from urllib.request import urlopen


def load_image_by_url(url):
    img = Image.open(urlopen(url))
    img.load()
    image_array = np.asarray(img, dtype="int32")
    return image_array


def load_image_by_path(path):
    img = Image.open(path)
    image_array = np.asarray(img, dtype="int32")
    return image_array


def save_image(np_data, output_filename) :
    img = Image.fromarray(np.asarray(np.clip(np_data, 0, 255), dtype="uint8"), "L" )
    img.save(output_filename)


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())