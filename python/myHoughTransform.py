import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    # Image dimensions
    W, H = img_threshold.shape
    # Diagonal length, maximum possible rho value
    rho_max = np.sqrt(H**2 + W**2)
    # Rho and Theta ranges
    rho_range = np.arange(0, rho_max, rhoRes)
    theta_range = np.arange(0, 2 * np.pi, thetaRes)

    # Cosine and sine computations
    cos_t = np.cos(theta_range)
    sin_t = np.sin(theta_range)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.float32)

    # (row, col) indexes to edges
    y_idxs, x_idxs = np.nonzero(img_threshold)

    # Vectorized computation for accumulator filling
    for x, y in zip(x_idxs, y_idxs):
        rho_values = x * cos_t + y * sin_t
        valid_rho_idxs = rho_values >= 0
        rho_idxs = (rho_values[valid_rho_idxs] / rhoRes).astype(int)
        theta_idxs = np.arange(len(theta_range))[valid_rho_idxs]

        np.add.at(accumulator, (rho_idxs, theta_idxs), 1)

    return accumulator, rho_range, theta_range
