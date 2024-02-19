import numpy as np
import cv2

def myHoughLines(H, nLines):
    kernel_size=(13,13)
    # Dilation for non-maximal suppression
    dilateKernel = np.ones(kernel_size, np.uint8)
    dilatedImg = cv2.dilate(H, dilateKernel)

    # Suppress non-maximal values
    H = np.where(H < dilatedImg, 0, H)

    # Get indeces of N largest values, from
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

    flattened = np.argpartition(H.flatten(), -nLines)[-nLines:]
    maxIndices = np.unravel_index(flattened, H.shape)

    # Sort by strength
    rhos, thetas = maxIndices
    strengths = H[rhos, thetas]
    sorted_indices = np.argsort(-strengths)

    # Extract and sort rhos and thetas
    sorted_rhos = rhos[sorted_indices]
    sorted_thetas = thetas[sorted_indices]

    return sorted_rhos, sorted_thetas
