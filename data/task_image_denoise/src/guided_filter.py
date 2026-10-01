# pip install opencv-contrib-python

import os, sys, cv2
from cv2.ximgproc import guidedFilter

ESC_KEY = 27

def callback(x):
    pass # do nothing

argv = sys.argv
argc = len(argv)

if(argc < 2):
    print('%s blurs the image with guided filter.' % argv[0])
    print('Usgae: >python %s <input image> [<guidance image>]' % argv[0])
    quit()

print('Hit ESC-key to terminate this program')

img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
#img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
#img = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
out = img.copy()

if argc > 2:
    guide = cv2.imread(argv[2])
else:
    guide = img

# create a window
cv2.namedWindow('image')

# create trackbar
cv2.createTrackbar('size', 'image', 1, 100, callback)
cv2.createTrackbar('sigma', 'image', 1, 1000, callback)
cv2.setTrackbarPos('sigma', 'image', 500)

prev_size = 1
prev_sigma = 500

print('Hit s-key to save and terminate')
print('Hit ESC-key to quit') 

while True:
    # retrieve the current position of trackbar
    size = cv2.getTrackbarPos('size', 'image') * 2 + 1 
    sigma = cv2.getTrackbarPos('sigma', 'image')

    # blur
    if(size != prev_size) or (sigma != prev_sigma):
        print('execute guided filter with the size %d sigma %d' % (size, sigma))
        out = cv2.ximgproc.guidedFilter(guide, img, size, sigma)
        prev_size = size
        prev_sigma = sigma

        cv2.imshow('image', out)

    key = cv2.waitKey(100)
    if key == ESC_KEY or key == ord('s') or key == ord('S'):
        break

cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_size%d_sigma%d.png' % (filename, size, sigma)

    cv2.imwrite(dst_path, out)
    print('save %s' % dst_path)