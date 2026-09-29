import numpy as np
import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s applies gamma to the image' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('original', src)

gamma = src.copy()
gamma = gamma.astype(np.float32) / 255.0
gamma **= 2.2

cv2.imshow('gamma', gamma)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    gamma *= 255
    gamma = np.clip(gamma, 0, 255)
    gamma = gamma.astype(np.uint8)

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_gamma.png' % filename

    cv2.imwrite(dst_path, gamma)
    print('save %s' % dst_path)

cv2.destroyAllWindows()
