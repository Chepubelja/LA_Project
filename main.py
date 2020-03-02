import os

from time import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *
from settings import *

from matplotlib import pyplot as plt

from wavelet_thresholding import denoise_image

from tqdm import tqdm


def evaluate(dataset_name: str, debug: bool = False):
    # Iterating over all given noise sigmas
    for sigma in SIGMAS:

        # Selecting dataset with appropriate noise level
        if dataset_name == 'BSD100':
            path_to_noisy_data = os.path.join(PATH_TO_BSD_100,
                                              'noisy{}'.format(sigma))
            path_to_original_data = os.path.join(PATH_TO_BSD_100,
                                                 'original_png')
        elif dataset_name == 'Urban100':
            path_to_noisy_data = os.path.join(PATH_TO_URBAN_100,
                                              'noisy{}'.format(sigma))
            path_to_original_data = os.path.join(PATH_TO_URBAN_100,
                                                 'original_png')
        else:
            raise AttributeError("Not appropriate dataset")

        # Preparing appropriate sigmas
        sigma = sigma / 100

        # Lists for gathering results
        psnr_results = {
            'VisuShrink': [],
            'BayesShrink': [],
            'SureShrink': []
        }
        ssim_results = {
            'VisuShrink': [],
            'BayesShrink': [],
            'SureShrink': []
        }
        time_results = {
            'VisuShrink': [],
            'BayesShrink': [],
            'SureShrink': []
        }

        # Iterating over all noisy images with the given noise level
        for image_name in tqdm(os.listdir(path_to_noisy_data),
                               position=0):

            # Loading image as numpy matrix
            noisy_image = load_image_by_path(os.path.join(path_to_noisy_data,
                                                          image_name))
            original_image = load_image_by_path(
                os.path.join(path_to_original_data, '_'.join(image_name.split('_')[:-2]) + '.png'))

            # Normalizing noisy and original images to range [0, 1]
            noisy_image_norm = normalize_image(noisy_image)
            original_image_norm = normalize_image(original_image)

            # Preparing noise sigmas for noisy image
            image_sigmas = [sigma / noisy_image_norm.ndim
                            for _ in range(noisy_image_norm.ndim)]

            # Iterating over all possible denoising methods
            for denoising_method in METHODS:

                # Calculating denoising time for the given method
                denoise_start_time = time()

                # Denoising image with the given method
                denoised_image = denoise_image(noisy_image_norm,
                                               sigma=image_sigmas,
                                               method=denoising_method,
                                               mode='soft')

                time_results[denoising_method] += [time() - denoise_start_time]

                denoised_image = np.clip(denoised_image, 0, 1)

                psnr = peak_signal_noise_ratio(denoised_image, original_image_norm)
                ssim = structural_similarity(denoised_image, original_image_norm, multichannel=True)

                # Storing results
                psnr_results[denoising_method] += [psnr]
                ssim_results[denoising_method] += [ssim]

                if debug:
                    fig = plt.figure(figsize=(14, 7))

                    fig.add_subplot(1, 2, 2)
                    plt.imshow(original_image_norm, interpolation='nearest')

                    fig.add_subplot(1, 2, 1)
                    plt.imshow(denoised_image)

                    fig.add_subplot(1, 2, 2)
                    plt.imshow(noisy_image_norm, interpolation='nearest')
                    # plt.show()
                    plt.savefig('results/BSD100/{}_{}.png'.format(image_name, denoising_method))

        print('\n===================')
        print('Sigma:', sigma)
        print('===================')

        for method in psnr_results.keys():
            print('Denoising method:', method)
            print('PSNR:', np.mean(psnr_results[method]))
            print('SSIM:', np.mean(ssim_results[method]))
            print('Time:', np.mean(time_results[method]))
            print()


if __name__ == '__main__':
    evaluate(dataset_name='BSD100', debug=True)
