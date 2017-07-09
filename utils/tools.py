import numpy as np
import math
import cv2

window_size_nnm = 10
initial_sigma = 1.5
filter_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
k = 0.04

# Sobel derivative 3x3 X
kernel_filter_dx_3 = np.float32(np.asarray([[-1, 0,  1],
                                            [-2, 0,  2],
                                            [-1, 0,  1]]))
kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]
kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]

# Sobel derivative 5x5 X
kernel_filter_dx_5 = np.float32(np.asarray([[  -1,  -2,  0,  2, 1],
                                              [-4,  -8,  0,  8, 4],
                                              [-6, -12,  0, 12, 6],
                                              [-4,  -8,  0,  8, 4],
                                              [-1,  -2,  0,  2, 1]]))
kernel_filter_dx_5 = kernel_filter_dx_5[..., np.newaxis]
kernel_filter_dx_5 = kernel_filter_dx_5[..., np.newaxis]

# Sobel derivative 3x3 Y
kernel_filter_dy_3 = np.float32(np.asarray([[-1, -2, -1],
                                            [ 0,  0,  0],
                                            [ 1,  2,  1]]))
kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]
kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]

# Sobel derivative 5x5 Y
kernel_filter_dy_5 = np.float32(np.asarray([[-1, -4,  -6, -4, -1],
                                              [-2, -8, -12, -8, -2],
                                              [ 0,  0,   0,  0,  0],
                                              [ 2,  8,  12,  8,  2],
                                              [ 1,  4,   6,  4,  1]]))
kernel_filter_dy_5 = kernel_filter_dy_5[..., np.newaxis]
kernel_filter_dy_5 = kernel_filter_dy_5[..., np.newaxis]




sobel3x = cv2.getDerivKernels(1, 0, 3, normalize=True)
sobel3x = np.outer(sobel3x[0], sobel3x[1])
sobel3x = sobel3x[..., np.newaxis]
sobel3x = sobel3x[..., np.newaxis]

sobel5x = cv2.getDerivKernels(1, 0, 5, normalize=True)
sobel5x = np.outer(sobel5x[0], sobel5x[1])
sobel5x = sobel5x[..., np.newaxis]
sobel5x = sobel5x[..., np.newaxis]

sobel7x = cv2.getDerivKernels(1, 0, 7, normalize=True)
sobel7x = np.outer(sobel7x[0], sobel7x[1])
sobel7x = sobel7x[..., np.newaxis]
sobel7x = sobel7x[..., np.newaxis]


sobel3y = cv2.getDerivKernels(0, 1, 3, normalize=True)
sobel3y = np.outer(sobel3y[0], sobel3y[1])
sobel3y = sobel3y[..., np.newaxis]
sobel3y = sobel3y[..., np.newaxis]

sobel5y = cv2.getDerivKernels(0, 1, 5, normalize=True)
sobel5y = np.outer(sobel5y[0], sobel5y[1])
sobel5y = sobel5y[..., np.newaxis]
sobel5y = sobel5y[..., np.newaxis]

sobel7y = cv2.getDerivKernels(0, 1, 7, normalize=True)
sobel7y = np.outer(sobel7y[0], sobel7y[1])
sobel7y = sobel7y[..., np.newaxis]
sobel7y = sobel7y[..., np.newaxis]













normfilter = np.ones([80,80])
normfilter = normfilter[..., np.newaxis]
normfilter = normfilter[..., np.newaxis]

normfilter_2 = np.ones([60,60])
normfilter_2 = normfilter_2[..., np.newaxis]
normfilter_2 = normfilter_2[..., np.newaxis]


def makeGaussian(size, center=None):
    # Make a square gaussian kernel.

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    index_sigma = filter_sizes.index(size)+1
    sigma = math.pow(initial_sigma, index_sigma)

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gaussian = np.float32((np.exp(-1 * (((x-x0)**2 + (y-y0)**2) / 2*(sigma**2)))) / (2*math.pi*sigma))
    gaussian = gaussian[..., np.newaxis]
    gaussian = gaussian[..., np.newaxis]

    return gaussian


def read_BW_Image(file_name):
    imageC = cv2.imread(file_name)
    # image = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image = cv2.cvtColor(imageC, cv2.COLOR_RGB2GRAY)
    # imageC = cv2.resize(imageC, (imageC.shape[0], imageC.shape[1]))
    # image = cv2.resize(image, (image.shape[0], image.shape[1]))
    if image is None:
        print('No image found:' + file_name)
        return None
    else:
        return imageC, image


gaussian_filter_5 = makeGaussian(5)