import cv2, os, sys
import numpy as np
import bm3d

sigma_255 = 20.0

argv = sys.argv
argc = len(argv)

print('%s denoises the image by Bilateral filter' % argv[0])
print('[usage] python %s <image> [<noise level (default:%.1f)>]' % (argv[0], sigma_255))

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('noisy', src)

noisy = src.copy()
noisy = noisy.astype(np.float32) / 255.0

if argc > 2:
    sigma_255 = float(argv[2])

sigma_psd = sigma_255 / 255.0

denoised = bm3d.bm3d(noisy, sigma_psd=sigma_psd)

denoised *= 255
denoised = np.clip(denoised, 0, 255)
denoised = denoised.astype(np.uint8)

cv2.imshow('denoised', denoised)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]

    noise_level = '%.0f' % sigma_255

    dst_path = '%s_denosed_by_bm3d_%s.png' % (filename, noise_level)
    cv2.imwrite(dst_path, denoised)
    print('save %s' % dst_path)

cv2.destroyAllWindows()
