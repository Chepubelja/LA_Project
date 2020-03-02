"""
Module with implementations
of 3 wavelet-based thresholds: Universal, Bayes and SURE.
"""

import numpy as np


def universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """
    return sigma * np.sqrt(2 * np.log(img.size))


def bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    # print("Thresh:", thresh)
    return thresh

def sure_threshold(image):
    """
    Adaptive Threshold Selection Using Principle of SURE
    """
    # thresholds = []

    print("Image shape:", image.shape)

    image = image.flatten()

    # for channel in range(image.ndim):
    #     image_channel = image[:, :, channel]
    n = np.size(image)
    a = np.sort(np.abs(image)) ** 2

    c = np.linspace(n - 1, 0, n)
    s = np.cumsum(a) + c * a
    risk = (n - (2 * np.arange(n)) + s)/n
    ibest = np.argmin(risk)
    threshold = np.sqrt(a[ibest])

    # thresholds += [threshold]

    # print(thresholds)
    # print(np.mean(thresholds))
    # return np.mean(thresholds)
    return threshold


def sure_threshold_bad(image, sigma):
    image = image.flatten()
    n = image.size
    print(n)
    # n = len(image)

    tmp = np.empty(shape=n)
    print(tmp)
    print(tmp.shape)
    print(sigma)
    # sigma = mad(data, n) / 0.6745;


    for i in range(n):
        tmp[i] = np.abs(image[i]) / sigma

    # for (i=0; i < n; i++)
    #     tmp[i] = fabs(data[i]) / sigma;

    suremin = np.inf
    tmp = sorted(tmp)
    # qsort(tmp, n, sizeof(double), abscmp)

    lambda_param = 0.0
    for k in range(n):
    # for (k=0; k < n; k++):
        sure = n - 2 * (k+1)+(n-k) * np.power(np.abs(tmp[k]), 2)
        for i in range(k):
        # for (i=0; i < k; i++):
            sure = sure + np.power(np.abs(tmp[i]), 2)
        if (sure < suremin):
            suremin = sure
            lambda_param = np.abs(tmp[k])
        print(sure, suremin)

    lambda_param = sigma * lambda_param
    return lambda_param


def thselect(x):
    # x = np.array(x)  # in case that x is not an array, convert it into an array
    x = x.flatten()
    l = len(x)

    sx2 = [sx * sx for sx in np.absolute(x)]

    sx2.sort()
    cumsumsx2 = np.cumsum(sx2)
    risks = []
    for i in range(0, l):
        risks.append((l - 2 * (i + 1) + (cumsumsx2[i] + (l - 1 - i) * sx2[i])) / l)
    mini = np.argmin(risks)
    th = np.sqrt(sx2[mini])
    print("Thresh:", th)
    return th


# def sure_shrink(coeffs, var):
#     N = len(coeffs)
#
#     sqr_coeffs = []
#     for coeff in coeffs:
#         sqr_coeffs.append(np.power(coeff, 2))
#
#     sqr_coeffs = np.concatenate(sqr_coeffs)
#
#     # print(sqr_coeffs)
#     sqr_coeffs.sort()
#     pos = 0
#     r = 0
#     from tqdm import tqdm
#
#     for idx, sqr_coeff in tqdm(enumerate(sqr_coeffs), position=0):
#         new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
#         if r == 0 or r > new_r:
#             r = new_r
#             pos = idx
#     thre = np.sqrt(var) * np.sqrt(sqr_coeffs[pos])
#     return thre


def sure_shrink(coeffs, var):
    # coeffs = np.concatenate(coeffs)
    # print(coeffs)
    # coeffs = [list(coef) for coef in coeffs]
    # print(coeffs)
    N = len(coeffs)
    print(N)
    # print(len(coeffs[0]))
    sqr_coeffs = []
    for coeff in coeffs:
        # print(np.sum(coeff))
        sqr_coeffs.append(np.power(coeff, 2))
    # sqr_coeffs = np.sort(sqr_coeffs)
    pos = 0
    r = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        print(idx)
        new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
        if r == 0 or r > new_r:
            r = new_r
            pos = idx
    thre = np.sqrt(var) * np.sqrt(sqr_coeffs[pos])
    return thre



def mini_max(coeffs, var):
    """Minimax threshold"""
    N = len(coeffs)
    if N > 32:
        return np.sqrt(var) * (0.3936 + 0.1829 * np.log2(N))
    else:
        return 0

import math
# import pywt
import bisect

def Sure_Shrink(coefficients):
    count = 0  # variable for try lambda
    minsure = 1e18  # variable to minimize the SURE equation
    t = 0  # the lambda that minimized SURE equation
    size = coefficients.shape  # get the size of coefficients
    # convert coefficients from 2D to 1D and put the absolute value in allcoeff
    numofcoefficients = size[0] * size[1]
    allcoeff = []
    com_sum = 0
    com_sum_coeff = []
    for listcoeff in coefficients:
        for coeff in listcoeff:
            allcoeff.append(abs(coeff))
    allcoeff.sort()  # sort this list
    for coeff in allcoeff:  # get the commulative sum of value^2 in allcoeff and put it in com_sum_coeff
        com_sum = com_sum + (abs(coeff) * abs(coeff))
        com_sum_coeff.append(com_sum)
    limit = math.sqrt(2 * math.log10(numofcoefficients))  # the limit for trying lambda
    while (count <= limit):
        sum = 0
        ind = bisect.bisect_right(allcoeff, count)  # get the index of abs(coeff)that is greater than count
        num_gt = len(com_sum_coeff) - ind  # number of abs(coeff)s that are greater than count
        sum = sum + (num_gt * count * count)  # apply the euation of SURE Shrink
        if (ind > 0):
            sum = sum + com_sum_coeff[ind - 1]
        sure = numofcoefficients + sum - 2 * ind
        if sure < minsure:  # Minimization condition for SURE equation
            minsure = sure
            t = count
        count = count + .01  # add small value to try more lambdas
    # coefficients = pywt.threshold(coefficients, t, 'soft')  # apply soft thrshold using lambda(t)

    # return coefficients
    return t


def sure_shrink_v2(coeffs):
    """
    http://www.csee.wvu.edu/~xinl/library/papers/math/statistics/donoho1995.pdf (1)
    :param coeffs: wavelet coeffs
    :return: optimal threshold
    """

    coeffs = coeffs.copy()
    coeffs = np.abs(coeffs.ravel())
    coeffs.sort()
    s = coeffs.size

    # noinspection PyTypeChecker
    sparse = 1/math.sqrt(s)*np.sum(coeffs**2 - 1)/math.log(s, 2)**(3/2)
    if sparse <= 1:
        return universal_threshold(coeffs)

    sure_min = np.inf
    optimal_thresh = 0
    num_smaller = 0
    sum_smaller = 0
    for thresh in coeffs:
        # Since thresh is increasing #coeffs < thresh will increase and can
        # keep track of these in coeffs_smaller and the bigger ones stay in coeffs array
        # For each iter we just look at the coeffs bigger than prev thresh. ca 100x faster
        # cardinality = coeffs[np.where(coeffs < thresh)].size
        # sure = s + sum(np.minimum(coeffs, thresh) ** 2) - 2 * cardinality

        coeffs_smaller = coeffs[np.where(coeffs < thresh)]
        coeffs = coeffs[np.where(coeffs >= thresh)]

        num_smaller += coeffs_smaller.size
        sum_smaller += np.sum(coeffs_smaller)
        sum_thresh = coeffs.size*thresh

        sure = s + (sum_smaller + sum_thresh)**2 - 2*num_smaller

        if sure < sure_min:
            sure_min = sure
            optimal_thresh = thresh

    return optimal_thresh


def universal_threshold(coeffs, sigma=1):
    if isinstance(coeffs, (int, float)):
        size = coeffs
    else:
        size = coeffs.size

    return np.sqrt(2*np.log(size))*sigma