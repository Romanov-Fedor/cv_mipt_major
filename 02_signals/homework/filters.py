import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    for i in range(Hi):
        for j in range(Wi):
            for k in range(-(Hk // 2), Hk // 2 + 1):
                for l in range(-(Wk // 2), Wk // 2 + 1):
                    if j + l >= 0 and i + k >= 0 and j + l < Wi and i + k < Hi:
                        out[i, j] += image[i + k, j + l] * \
                            kernel[Hk//2 - k, Wk//2 - l]

    # YOUR CODE HERE
    # END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + pad_height * 2, W + pad_width * 2))
    out[pad_height: H + pad_height, pad_width: W + pad_width] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    true_kernel = np.flip(kernel)
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i: i + Hk, j: j + Wk]
            out[i, j] = np.sum(region * true_kernel)
    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    def _circular_extension_2d(kernel, num_rows, num_cols):
        kernel_radius_v = kernel.shape[0] // 2
        kernel_radius_h = kernel.shape[1] // 2

        kernel_padded = np.zeros((num_rows, num_cols), dtype=kernel.dtype)
        kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
        kernel_padded = np.roll(
            kernel_padded, shift=(-kernel_radius_v, -kernel_radius_h), axis=(0, 1)
        )
        return kernel_padded

    Hi, Wi = image.shape

    padded_image = np.zeros((max(Hi, Wi) + 1, max(Hi, Wi) + 1))
    padded_image[:Hi, :Wi] = image
    padded_kernel = _circular_extension_2d(
        kernel, max(Hi, Wi) + 1, max(Hi, Wi) + 1)

    f_image = np.fft.fft2(padded_image)
    f_kernel = np.fft.fft2(padded_kernel)
    f_out = f_image * f_kernel
    out = np.real(np.fft.ifft2(f_out))

    return out[:Hi, :Wi]


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_faster(f, np.flip(g))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_faster(f, np.flip(g - np.mean(g)))


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    padded_image = zero_pad(f, Hk // 2, Wk // 2)

    g = (g - np.mean(g)) / np.std(g)
    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i: i + Hk, j: j + Wk]
            region = (region - np.mean(region)) / np.std(region)
            out[i, j] = np.sum(region * g)

    return out
