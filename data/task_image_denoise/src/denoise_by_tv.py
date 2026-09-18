import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle
from skimage.util import random_noise

import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s denoise image by TV' % argv[0])
print('[usage] python %s <image> [<weight(default: 0.05)>]' % argv[0])

if argc < 2:
    quit()

# 1. 元画像の準備（浮動小数点数 [0, 1] に変換）

src_img = cv2.imread(argv[1])
cv2.imshow('source', src_img)

weight = 0.05
if argc > 2:
    weight = float(argv[2])

bgr_img = src_img.copy()
bgr_img = bgr_img.astype(np.float32) / 255.0

b_img, g_img, r_img = cv2.split(bgr_img)

# 3. Total Variation デノイズの適用
# weight（λの逆数に対応）が大きいほど、平滑化（ノイズ除去）が強くなります
# 通常は 0.05 から 0.2 あたりで調整します
denoised_b = denoise_tv_chambolle(b_img, weight=weight)
denoised_g = denoise_tv_chambolle(g_img, weight=weight)
denoised_r = denoise_tv_chambolle(r_img, weight=weight)

denoised_bgr = cv2.merge([denoised_b, denoised_g, denoised_r])
denoised_bgr *= 255.0

denoised_bgr = np.clip(denoised_bgr, 0, 255)
denoised_bgr = denoised_bgr.astype(np.uint8)

cv2.imshow('denosed', denoised_bgr)

print('Hit s-key to save the result and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):
    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    w = '%.2f' % weight
    w = w.replace('.', '_')
    dst_path = '%s_denoised_by_tv%s.png' % (filename, w)
    cv2.imwrite(dst_path, denoised_bgr)
    print('save %s' % dst_path)

cv2.destroyAllWindows()

