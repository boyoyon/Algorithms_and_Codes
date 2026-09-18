import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import data, img_as_float
import numpy as np

import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s denoises the image by wavelet shrinkage' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('noisy', src)

noisy = src.copy() 
noisy = noisy.astype(np.float32) / 255.0

# ノイズの標準偏差を推定し、ウェーブレットデノイズを実行（BayesShrink法）
sigma_est = estimate_sigma(noisy, channel_axis=-1)
denoised = denoise_wavelet(noisy, method='BayesShrink', mode='soft', 
                           wavelet='db1', channel_axis=-1)

denoised *= 255.0
denozed = np.clip(denoised, 0, 255.0)
denoised = denoised.astype(np.uint8)

cv2.imshow('denoised', denoised)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_denosed_by_wavelet_shrinkage.png' % filename
    cv2.imwrite(dst_path, denoised)
    print('save %s' % dst_path)

cv2.destroyAllWindows()