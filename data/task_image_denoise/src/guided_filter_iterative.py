# pip install opencv-contrib-python

import os, sys, cv2
from cv2.ximgproc import guidedFilter

size = 15
sigma = 50

font = cv2.FONT_HERSHEY_PLAIN
font_size = 2
font_color = (0, 255, 0)
font_pos = (10, 30)

argv = sys.argv
argc = len(argv)

print('%s blurs the image with guided filter.' % argv[0])
print('[usage] python %s <input image> <guidance image>[<size(%d)> <sigma(%d)>]' % (argv[0], size, sigma))

if argc < 3:
    quit()

img = cv2.imread(argv[1])
cv2.imshow('input', img)

guide = cv2.imread(argv[2])
cv2.imshow('output', guide)

if argc > 3:
   size = int(argv[3])
   if size % 2 == 0:
       size += 1

if argc > 4:
    sigma = int(argv[4])

print('Hit any key to terminate this program')

no = 1

while True:
    guide = cv2.ximgproc.guidedFilter(guide, img, size, sigma)

    clone = guide.copy()
    cv2.putText(clone, '%d' % no, font_pos, font, font_size, font_color, 2)
    cv2.imshow('output', clone)
    no += 1

    key = cv2.waitKey(10)

    if key != -1:
        break

cv2.destroyAllWindows()
dst_path = 'guided_filtered_size%d_sigma%d_iterate%d.png' % (size, sigma, no)
cv2.imwrite(dst_path, guide)
print('save %s' % dst_path)
