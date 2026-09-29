import numpy as np
import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s applies inverse gamma to the image' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('original', src)

inv_gamma = src.copy()
inv_gamma = inv_gamma.astype(np.float32) / 255.0
inv_gamma **= (1/2.2)

cv2.imshow('inverse gamma', inv_gamma)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    inv_gamma *= 255
    inv_gamma = np.clip(inv_gamma, 0, 255)
    inv_gamma = inv_gamma.astype(np.uint8)

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_inv_gamma.png' % filename

    cv2.imwrite(dst_path, inv_gamma)
    print('save %s' % dst_path)

cv2.destroyAllWindows()
