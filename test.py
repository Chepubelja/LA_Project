from wavelet_thresholding import denoise_image

from utils import *

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from matplotlib import pyplot as plt


if __name__ == '__main__':

    test_image_link = "https://user-images.githubusercontent.com/22610398/75276514-f451a800-580e-11ea-9f9e-c1539b93e726.png"
    test_image = load_image_by_url(test_image_link)
    # print(test_image.shape)

    # plt.figure(figsize=(12, 7))
    # plt.imshow(test_image, interpolation='nearest')
    # plt.show()

    test_image = normalize_image(test_image)
    #
    # print(test_image.min(), test_image.max())
    #
    # # test_image = img_as_float(test_image)
    # # print(test_image.min(), test_image.max())

    # test_image_gray = rgb2gray(test_image)

    # print(test_image_gray.min(), test_image_gray.max())
    # # test_image_gray += 0.1 * np.random.randn(*test_image_gray.shape)
    # test_image_gray = np.clip(test_image_gray, 0, 1)
    # print(test_image_gray.min(), test_image_gray.max())
    # print(test_image_gray.shape)
    #
    # sigma_est = estimate_sigma(test_image_gray, multichannel=False, average_sigmas=True)
    # print(sigma_est)
    #
    # denoised_img = denoise_wavelet(test_image_gray, sigma=sigma_est,
    #                                rescale_sigma=True, method='BayesShrink')
    # print(denoised_img.shape)
    #
    # plt.figure(figsize=(12, 7))
    # plt.imshow(denoised_img, interpolation='nearest', cmap='gray')
    # plt.show()

    # Estimate the average noise standard deviation across color channels.
    # sigma_est = estimate_sigma(test_image, multichannel=True, average_sigmas=False)
    # print(sigma_est)

    result = denoise_image(test_image, sigma = [0.25 / 3 for _ in range(test_image.ndim)], method = 'SureShrink')
    result = np.clip(result, 0, 1)
    # result *= 255
    print(result)
    print(result.min(), result.max())

    psnr = peak_signal_noise_ratio(result, test_image)
    ssim = structural_similarity(result, test_image, multichannel=True)

    # print('Denoising method:', denoising_method)
    # print('Sigma:', sigma)
    print('PSNR:', psnr)
    print('SSIM:', ssim)
    print()

    # print(result.shape)
    # print(result.min(), result.max())

    fig = plt.figure(figsize=(14, 7))

    fig.add_subplot(1, 2, 1)
    plt.imshow(test_image)

    fig.add_subplot(1, 2, 2)
    plt.imshow(result, interpolation='nearest', cmap='gray')
    plt.show()