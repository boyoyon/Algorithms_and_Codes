import numpy as np
import cv2, os, sys

SIZE = 3

argv = sys.argv
argc = len(argv)

print('%s denoises the image by gaussian filter' % argv[0])
print('[usage] python %s <image> [<size (default:3)>]' % argv[0])

if argc < 2:
    quit()

noisy = cv2.imread(argv[1])
cv2.imshow('noisy', noisy)

if argc > 2:
    SIZE = int(argv[2])
    if SIZE % 2 == 0:
        SIZE += 1

denoised = cv2.GaussianBlur(noisy, (SIZE, SIZE), 0)
cv2.imshow('denoised', denoised)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_denosed_by_gaussian%03d.png' % (filename, SIZE)
    cv2.imwrite(dst_path, denoised)
    print('save %s' % dst_path)

cv2.destroyAllWindows()