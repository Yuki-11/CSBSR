##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This code is based on https://github.com/XPixelGroup/BasicSR, and is licensed Apache LICENSE.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import numpy as np
import glob

from metrics_util import reorder_image, to_y_channel

class BaseMetrix:
    def __init__(self, input_order='BHWC', test_y_channel=True, standardized = True):
        if input_order not in ['BHWC', 'BCHW']:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are '
                '"BHWC" and "BCHW"')
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.standardized = standardized

    def pre_process(self, crop_border):
        assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
        
        if standardized:
            img1, img2 = list(map(lambda x: np.round(255 * x).astype(np.int64), [img1, img2]))
        img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
        img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

        if crop_border != 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        return img1, img2


class PSNR(BaseMetrix):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """
    def __init__(self, input_order='BHWC', test_y_channel=True, standardized=True):
        
        super().__init__(input_order, test_y_channel, standardized)

    def calc(self, img1, img2):
        img1, img2 = self.pre_process(img1, img2)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))


class SSIM(BaseMetrix):
    """
    Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """
    def __init__(self, input_order='BHWC', test_y_channel=True, standardized=True):
        super().__init__(input_order, test_y_channel, standardized)

    def calc(self, img1, img2):
        img1, img2 = self.pre_process(img1, img2)
        """Calculate SSIM (structural similarity) for one channel images.
        It is called by func:`calculate_ssim`.
        Args:
            img1 (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.
        Returns:
            float: ssim result.
        """

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))


def calculate_psnr(img1, img2, crop_border, input_order='BHWC', test_y_channel=True, standardized = True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['BHWC', 'BCHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"BHWC" and "BCHW"')
    if standardized:
        img1, img2 = list(map(lambda x: np.round(255 * x).astype(np.int64), [img1, img2]))
    img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=True):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def main():
    test_path = '/home/mori/TecoGAN-PyTorch/results/Vid4/TecoGAN_BI_iter500000/calendar/'
    gt_path = '/home/mori/TecoGAN-PyTorch/data/Vid4/GT/calendar/'
    test_files = glob.glob(test_path + '*')
    gt_files = glob.glob(gt_path + '*')
    sum_psnr = 0
    sum_ssim = 0
    length = len(test_files)
    for (test_file, gt_file) in zip(test_files, gt_files):
        test_image = cv2.imread(test_file)
        gt_image = cv2.imread(gt_file)
        psnr = calculate_psnr(test_image, gt_image, 4)
        ssim = calculate_ssim(test_image, gt_image, 4)
        #print(psnr)
        sum_psnr += psnr
        sum_ssim += ssim
    print(test_path)
    print("averaged psnr {}".format(sum_psnr/length))
    print("averaged ssim {}".format(sum_ssim/length))


if __name__ == '__main__':
    main()
