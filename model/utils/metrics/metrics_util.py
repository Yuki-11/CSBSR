##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This code is based on https://github.com/XPixelGroup/BasicSR, and is licensed Apache LICENSE.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

from matlab_functions import bgr2ycbcr

def reorder_image(img, input_order='BHWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (b, h, w), return (b, h, w, 1);
    If the input_order is (b, c, h, w), return (b, h, w, c);
    If the input_order is (b, h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'BHWC' or 'BCHW'.
            If the input image shape is (b, h, w), input_order will not have
            effects. Default: 'BHWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['BHWC', 'BCHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'BHWC' and 'BCHW'")
    if len(img.shape) != 4:
        additional_dim = [1 for _ in range(4 - len(img.shape))]
        img = img.reshape(*additional_dim, *img.shape)
    if input_order == 'BCHW':
        img = img.transpose(0, 2, 3, 1)
    
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 4 and img.shape[3] == 3:
        img = rgb2ycbcr(img, y_only=True)
        # img = img[..., None]
    return img * 255.
