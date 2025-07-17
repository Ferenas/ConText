import numpy as np
import math
from scipy import signal, ndimage

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (0x44, 0x01, 0x54)
YELLOW = (0xFD, 0xE7, 0x25)

from sklearn.metrics import confusion_matrix





class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def F_Score(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP 
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)
        return P, R, F

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoUs = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoUs

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image,ignore_value=None):
        assert gt_image.shape == pre_image.shape
        gt_mask = np.all(gt_image == [255, 255, 255], axis=-1).astype(int)  # GT: 1 为前景，0 为背景
        pred_mask = np.all(pre_image == [255, 255, 255], axis=-1).astype(int)
        if ignore_value != None:
            gt_mask = gt_mask.flatten()
            pred_mask = pred_mask.flatten()
            valid_indices = gt_mask != ignore_value
            pred_mask = pred_mask[valid_indices]
            gt_mask = gt_mask[valid_indices]
            self.confusion_matrix += confusion_matrix(gt_mask.flatten(), pred_mask.flatten())
        else:
            self.confusion_matrix += confusion_matrix(gt_mask.flatten(), pred_mask.flatten())
        # self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
    def print_result(self, task):
        P, R, F = self.F_Score()
        IoUs = self.Mean_Intersection_over_Union()
        
        if task == 'text segmentation':
            print(f'fgIoU: {IoUs[1]}; P: {P[1]}; R: {R[1]}; F: {F[1]}')
        elif task == 'tampered text detection':
            print('Real Text:')
            print(f'  IoU: {IoUs[1]}; P: {P[1]}; R: {R[1]}; F: {F[1]}')
            print('Tampered Text:')
            print(f'  IoU: {IoUs[2]}; P: {P[2]}; R: {R[2]}; F: {F[2]}')
            print('Average:')
            print(f'  mIoU: {np.nanmean(IoUs[-2:])}; mF: {np.nanmean(F[-2:])}')
        else:
            raise ValueError



def seed_everything(seed):
    import random, os, torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metric(args, target, ours):
    # Crop the right area:
    return _calc_metric(ours, target)


def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array
    
    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where
    
    A = 1/(2*pi*sigma^2)
    
    as define by Wolfram Mathworld 
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()



def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
  #  import pdb;pdb.set_trace()
    mu1 = signal.fftconvolve(img1, window, mode = 'valid')
    mu2 = signal.fftconvolve(img2, window, mode = 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode = 'valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode = 'valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode = 'valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map = True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(img1, downsample_filter, 
                                                mode = 'reflect')
        filtered_im2 = ndimage.filters.convolve(img2, downsample_filter, 
                                                mode = 'reflect')
        im1 = filtered_im1[: : 2, : : 2]
        im2 = filtered_im2[: : 2, : : 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0 : level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0 : level - 1]), weight[0 : level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power


def _calc_metric_generation(ours,target,mask):
    # count_ones = np.sum(mask == 255)
    ours,target = ours / 255, target /255
    ours = ours.transpose(2,0,1)
    target = target.transpose(2,0,1)
    
    mse = ((ours - target)**2).mean()
    # if mse < 1e-10:
    #     psnr = 100
    # else:
    #     psnr = 10 * math.log10(1/(mse))
    psnr = 10 * math.log10(1/(mse))

    R = target[0,:, :]
    G = target[1,:, :]
    B = target[2,:, :]
    YGT = .299 * R + .587 * G + .114 * B

    R = ours[0,:, :]
    G = ours[1,:, :]
    B = ours[2,:, :]
    YBC = .299 * R + .587 * G + .114 * B
    Diff = abs(np.array(YBC*255) - np.array(YGT*255)).round().astype(np.uint8)
    AGE = np.mean(Diff)
    mssim = msssim(np.array(YGT*255), np.array(YBC*255))
    

    threshold = 20  #ScuSyn 35
    Errors = Diff > threshold
    EPs = sum(sum(Errors)).astype(float)
    pEPs = EPs / float(ours.shape[1]*ours.shape[2])

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    erodedErrors = ndimage.binary_erosion(Errors, structure).astype(Errors.dtype)
    CEPs = sum(sum(erodedErrors))
    pCEPs = CEPs / float(ours.shape[1]*ours.shape[2])

    return np.array([mse,psnr,AGE,mssim,pEPs,pCEPs])
    # mse = ((ours/255 - target/255)**2).sum() / (count_ones + 1e-6)
    # return {'mse':mse,'psnr':psnr,'age':AGE,'mssim':mssim,'peps':pEPs,'pceps':pCEPs}

def _calc_metric(ours, target, fg_color=WHITE, bg_color=BLACK):
    fg_color = np.array(fg_color)
    # Calculate accuracy:
    assert target.shape == ours.shape
    accuracy = np.sum(np.float32((target == ours).all(axis=2))) / (ours.shape[0] * ours.shape[1])
    seg_orig = ((target - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    seg_our = ((ours - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    color_blind_seg_our = (ours - np.array([[bg_color]]) != 0).any(axis=2)
    iou = np.sum(np.float32(seg_orig & seg_our)) / (np.sum(np.float32(seg_orig | seg_our)) + 1e-6)
    # color_blind_iou = np.sum(np.float32(seg_orig & color_blind_seg_our)) / np.sum(
    #     np.float32(seg_orig | color_blind_seg_our))
    return {'iou': iou, 'accuracy': accuracy}


# def get_default_mask_1row_mask():
#     mask = np.zeros((14,14))
#     mask[:7] = 1
#     mask[:, :7] = 1
#     return mask