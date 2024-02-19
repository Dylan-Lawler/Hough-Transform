import numpy as np

def myImageFilter(img, h):
    # Ensure the image is a float64 array
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    # Get the dimensions of the image and the kernel
    height, width = img.shape
    kernel_height, kernel_width = h.shape

    # Calculate the necessary padding for height and width
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image using 'edge' mode
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    # Initialize an array to store the filtered image
    filtered_img = np.zeros_like(img)

    # Create an array of the indices of the original image
    i, j = np.indices((height, width))

    # Vectorized convolution operation
    for y in range(kernel_height):
        for x in range(kernel_width):
            # Apply kernel to the corresponding region of the image
            filtered_img += h[y, x] * padded_img[i + y, j + x]

    return filtered_img
