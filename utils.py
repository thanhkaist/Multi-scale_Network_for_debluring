import os
import os.path
import torch
import sys
from functools import reduce
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image
import numpy as np
import skvideo.measure.msssim as msssim

class SaveData():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
            self.logCsv = open(self.save_dir + '/log.csv', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
            self.logCsv = open(self.save_dir + '/log.csv', 'w')
        self.best_score = 0

    def save_model(self, model, epoch, score):
        if self.args.multi:
            model = model.module
        torch.save(model.state_dict(), self.save_dir_model + '/model_lastest.pt')
        torch.save(model.state_dict(), self.save_dir_model + '/model_' + str(epoch) + '.pt')
        torch.save(model, self.save_dir_model + '/model_obj.pt')
        torch.save(epoch, self.save_dir_model + '/last_epoch.pt')
        if score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.save_dir_model + '/model_best.pt')
            torch.save(epoch, self.save_dir_model + '/best_epoch.pt')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()

    def load_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
        last_epoch = torch.load(self.save_dir_model + '/last_epoch.pt')
        print("load mode_status frmo {}/model_lastest.pt, epoch: {}".format(self.save_dir_model, last_epoch))
        return model, last_epoch

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_dir_model + '/model_best.pt'))
        best_epoch = torch.load(self.save_dir_model + '/best_epoch.pt')
        print("load mode_status frmo {}/model_best.pt, epoch: {}".format(self.save_dir_model, best_epoch))
        return model, best_epoch

    def write_csv_header(self,*args):
        self.log_csv(*args)

    def log_csv(self,*args):
        log = ""
        sys.stdout.flush()
        for i in args:
            log += str(i)+','
        self.logCsv.write(log[:-1]+'\n')
        self.logCsv.flush()

class AverageMeter():
    __var = []
    __sum = 0.0
    __avg = 0.0
    __count = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.__var.clear()
        self.__sum = 0
        self.__avg = 0
        self.__count = 0

    def update(self, val, n=1):
        self.__var.extend([val] * n)
        self.__count += n
        self.__sum += val * n

    def val(self):
        if len(self.__var) == 0:
            return None
        return self.__var[-1]

    def avg(self):
        if self.__count == 0:
            return None
        return self.__sum / self.__count

    def sum(self):
        return self.__sum

    def reduce_avg(self):
        if self.__count == 0:
            return None
        return reduce(lambda x, y: x + y, self.__var) / len(self.__var)


def unnormalize(gpu_tensor):
    out = gpu_tensor.data.cpu().numpy()

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    #for t, m, s in zip(out, mean, std):
    #    t.mul_(s).add_(m)

    nor = out*255.0
    nor = nor.clip(0, 255)

    nor = nor.transpose(1, 2, 0) 
    return nor


def psnr_ssim_from_sci(img1, img2, padding=4, y_channels = False):
    '''
    Calculate PSNR and SSIM on Y channels for image super resolution
    :param img1: numpy array
    :param img2: numpy array
    :param padding: padding before calculate
    :return: psnr, ssim
    '''

    img1 = Image.fromarray(np.uint8(img1), mode='RGB')
    img2 = Image.fromarray(np.uint8(img2), mode='RGB')
    if y_channels:
        img1 = img1.convert('YCbCr')
        img1 = np.ndarray((img1.size[1], img1.size[0], 3), 'u1', img1.tobytes())

        img2 = img2.convert('YCbCr')
        img2 = np.ndarray((img2.size[1], img2.size[0], 3), 'u1', img2.tobytes())
        # get channel Y
        img1 = img1[:, :, 0]
        img2 = img2[:, :, 0]
        # padding
        img1 = img1[padding: -padding, padding:-padding]
        img2 = img2[padding: -padding, padding:-padding]
        ss = ssim(img1, img2)
        ps = psnr(img1, img2)
    else:
        # padding
        img1 = np.array(img1)
        img2 = np.array(img2)
        # img1 = img1[padding: -padding, padding:-padding,:]
        # img2 = img2[padding: -padding, padding:-padding,:]
        ps = psnr(img1,img2,255)
        ss = ssim(img1,img2,multichannel=True)

    return (ps, ss)


def msssim_from_images(img1, img2):
    # get luminance channel
    img1 = Image.fromarray(np.uint8(img1), mode='RGB')
    img2 = Image.fromarray(np.uint8(img2), mode='RGB')
    img1 = img1.convert('YCbCr')
    img1 = np.ndarray((img1.size[1], img1.size[0], 3), 'u1', img1.tobytes())

    img2 = img2.convert('YCbCr')
    img2 = np.ndarray((img2.size[1], img2.size[0], 3), 'u1', img2.tobytes())
    # get channel Y
    img1 = img1[:, :, 0]
    img2 = img2[:, :, 0]
    _msssim = msssim(img1, img2)[0]
    return _msssim


def output_psnr_mse(img_orig, img_out,max_val = 1.0):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(max_val / mse)
    return psnr


## Multiscale ssim function
# refer from
# https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve

def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
  im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
  mssim = np.array([])
  mcs = np.array([])
  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
  return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))