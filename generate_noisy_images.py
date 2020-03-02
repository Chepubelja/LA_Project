"""
Module for preparing BSD100 and Urban100 datasets.
"""

import os

# from PIL import Image
from skimage.util import random_noise

from utils import *
from settings import *

import matplotlib.pyplot as plt

from tqdm import tqdm

def generate_noise_for_dataset(dataset_name, dataset_images_path):

    print("Dataset:", dataset_name)

    for sigma in SIGMAS:

        print("Sigma:", sigma)

        path_to_noisy = os.path.join('data', dataset_name, 'noisy{}'.format(sigma))

        if not os.path.exists(path_to_noisy):
            os.makedirs(path_to_noisy)

        dataset_images_path_upper = '/'.join(dataset_images_path.split('/')[:2])
        path_to_images =  os.path.join(dataset_images_path_upper, 'noisy{}'.format(sigma))

        for image_name in tqdm(os.listdir(dataset_images_path), position=0):
            original_image = load_image_by_path(os.path.join(dataset_images_path, image_name))
            plt.imshow(original_image, interpolation='nearest')
            original_image = normalize_image(original_image)
            # print(original_image.min(), original_image.max())

            noisy_image = random_noise(original_image, mode='gaussian',
                                        mean=0, var=(sigma / 100 / 3) ** 2)
            # print(noisy_image.shape)

            # fig = plt.figure(figsize=(14, 7))

            # fig.add_subplot(1, 2, 1)
            # plt.imshow(noisy_image)
            #
            # fig.add_subplot(1, 2, 2)
            # plt.imshow(original_image, interpolation='nearest')
            # plt.show()

            # pil_im = Image.fromarray(noisy_image)
            import scipy.misc

            image_name_without_ext = image_name.split('.')[0]
            # pil_im.save(os.path.join(path_to_images, "{}_sigma_{}.png".format(image_name_without_ext, sigma)))
            scipy.misc.imsave(os.path.join(path_to_images,
                                           "{}_sigma_{}.png".format(image_name_without_ext,
                                                                    sigma)),
                              noisy_image)


if __name__ == '__main__':
    generate_noise_for_dataset(dataset_name='BSD100', dataset_images_path=os.path.join(PATH_TO_BSD_100, 'original_png'))
    generate_noise_for_dataset(dataset_name='URBAN100', dataset_images_path=os.path.join(PATH_TO_URBAN_100, 'original_png'))
