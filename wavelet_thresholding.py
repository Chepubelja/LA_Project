import pywt

from scipy.stats import norm
from utils import normalize_image

from thresholds import *


def scale_sigma_and_image_consistently(image, sigma):
    """If the ``image`` is rescaled, also rescale ``sigma`` consistently.
    Images that are not floating point will be rescaled via ``img_as_float``.
    """
    if image.dtype.kind != 'f':
        range_pre = image.max() - image.min()

        # Normalizing image to 0-1 (floats) instead 0-255
        image = normalize_image(image)
        range_post = image.max() - image.min()

        # apply the same magnitude scaling to sigma
        scale_factor = range_post / range_pre
        if image.ndim > 1:
            sigma = [s * scale_factor if s is not None else s
                     for s in sigma]
        elif sigma is not None:
            sigma *= scale_factor
    return image, sigma


def calculate_threshold(method, image, sigma, dcoeffs):

    if method == "BayesShrink":
        # print(2)
        var = sigma ** 2
        # The BayesShrink thresholds from [1]_ in docstring
        threshold = [{key: bayes_thresh(level[key], var) for key in level}
                     for level in dcoeffs]

    elif method == "VisuShrink":
        # print(1)
        # The VisuShrink thresholds from [2]_ in docstring
        threshold = universal_thresh(image, sigma)

    elif method == "SureShrink":
        # print(len(dcoeffs))
        # print(len(dcoeffs[0]))
        # print(dcoeffs[0])
        # var = sigma ** 2
        # threshold = [{key: Sure_Shrink(level[key]) for key in level}
        #              for level in dcoeffs]

        threshold = [{key: sure_shrink_v2(level[key]) for key in level}
                     for level in dcoeffs]
        # print(threshold)

    elif method == "Minimax":
        var = sigma ** 2
        threshold = [{key: mini_max(level[key], var) for key in level}
                     for level in dcoeffs]

    return threshold


def estimate_sigma(detail_coeffs):
    # 75th quantile of the underlying, symmetric noise distribution
    denom = norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom
    return sigma


def wavelet_threshold(image, wavelet_type, method, sigma, mode='soft'):
    wavelet = pywt.Wavelet(wavelet_type)

    original_extent = tuple(slice(s) for s in image.shape)

    dlen = wavelet.dec_len
    wavelet_levels = np.min(
        [pywt.dwt_max_level(s, dlen) for s in image.shape])

    # Skip coarsest wavelet scales.
    if method != 'SureShrink':
        wavelet_levels = max(wavelet_levels - 3, 1)
    else:
        wavelet_levels = 1

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    # Computing the selected threshold
    threshold = calculate_threshold(method, image, sigma, dcoeffs)

    if np.ndim(threshold) == 0:
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]

    denoised_coeffs = [coeffs[0]] + denoised_detail

    # STEP 3: Inverse Discrete Wavelet Transform.
    inverted_wavelet = pywt.waverecn(denoised_coeffs, wavelet)[original_extent]
    return inverted_wavelet


def denoise_image(image, sigma, method, wavelet_type='db1', mode='soft'):

    # image, sigma = scale_sigma_and_image_consistently(image, sigma)

    if image.ndim > 1:
        denoised_image = np.empty_like(image)
        for c in range(image.shape[-1]):
            denoised_image[..., c] = wavelet_threshold(image[..., c],
                                            wavelet_type=wavelet_type,
                                             method=method,
                                             sigma=sigma[c], mode=mode)

    else:
        denoised_image = wavelet_threshold(image, wavelet_type=wavelet_type, method=method,
                                 sigma=sigma, mode=mode)

    return denoised_image
