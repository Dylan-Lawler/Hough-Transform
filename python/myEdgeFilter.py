import numpy as np
import scipy.signal
from math import ceil
from myImageFilter import myImageFilter
import cv2

def myEdgeFilter(img0, sigma):
    # Gaussian kernel
    hsize = int(2 * ceil(3 * sigma) + 1)
    gaussian_kernel = scipy.signal.windows.gaussian(hsize, sigma)
    gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel)
    smooth_img = myImageFilter(img0, gaussian_kernel)
    
    # Sobel filters for x and y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    # Gradient computation
    imgx = myImageFilter(smooth_img, sobel_x)
    imgy = myImageFilter(smooth_img, sobel_y)

    # Edge magnitude
    magnitude = np.sqrt(imgx ** 2 + imgy ** 2)

    # Normalize magnitude to range [0, 1]
    magnitude = magnitude / magnitude.max()

    # Compute the angle of gradients
    angle = np.arctan2(imgy, imgx) * 180 / np.pi
    angle[angle < 0] += 180

    # Padding the magnitude array for border handling
    magnitude_padded = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Define the angle thresholds
    angle_0 = ((angle >= 157.5) | (angle < 22.5))
    angle_45 = ((angle >= 22.5) & (angle < 67.5))
    angle_90 = ((angle >= 67.5) & (angle < 112.5))
    angle_135 = ((angle >= 112.5) & (angle < 157.5))

    # Vectorized non-maximum suppression
    # Extract neighboring magnitudes in all four directions
    neighbor_0 = np.maximum(magnitude_padded[1:-1, :-2], magnitude_padded[1:-1, 2:])
    neighbor_45 = np.maximum(magnitude_padded[:-2, 2:], magnitude_padded[2:, :-2])
    neighbor_90 = np.maximum(magnitude_padded[:-2, 1:-1], magnitude_padded[2:, 1:-1])
    neighbor_135 = np.maximum(magnitude_padded[:-2, :-2], magnitude_padded[2:, 2:])

    # Suppress non-maximum
    magnitude_suppressed = np.zeros_like(magnitude)
    magnitude_suppressed[angle_0] = magnitude[angle_0] * (magnitude[angle_0] >= neighbor_0[angle_0])
    magnitude_suppressed[angle_45] = magnitude[angle_45] * (magnitude[angle_45] >= neighbor_45[angle_45])
    magnitude_suppressed[angle_90] = magnitude[angle_90] * (magnitude[angle_90] >= neighbor_90[angle_90])
    magnitude_suppressed[angle_135] = magnitude[angle_135] * (magnitude[angle_135] >= neighbor_135[angle_135])

    return magnitude_suppressed
