import os, sys, cv2
# import numpy as np
import copy

def callback(x):
    pass # do nothing

argvs = sys.argv
argc = len(argvs)

if(argc < 2):
    print('%s blurs the image with gaussian filter.' % argvs[0])
    print('Usgae: >python %s <input image> [<output image>]' % argvs[0])
    quit()

print('Hit ESC-key to terminate this program')

#img = cv2.imread(argvs[1], cv2.IMREAD_COLOR)
#img = cv2.imread(argvs[1], cv2.IMREAD_GRAYSCALE)
img = cv2.imread(argvs[1], cv2.IMREAD_UNCHANGED)
out = copy.copy(img)

# create a window
cv2.namedWindow('image')

# create trackbar
cv2.createTrackbar('size', 'image', 1, 1000, callback)

prev_size = 1

while(1):
    # retrieve the current position of trackbar
    size = cv2.getTrackbarPos('size', 'image') * 2 + 1 

    # blur with gaussian
    if(size != prev_size):
        print('execute gaussian filter with the size %d' % size)
        out = cv2.GaussianBlur(img,(size, size), 0)
        prev_size = size

    cv2.imshow('image',out)

    key = cv2.waitKey(100)
    if(key == 27):
        break

dst_path = 'gauss_%d.png' % size
cv2.imwrite(dst_path, out)
print('save %s' % dst_path)

cv2.destroyAllWindows()
