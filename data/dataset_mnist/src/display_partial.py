import cv2, sys
import numpy as np

NR_IMAGES = 10
WIDTH = 28
HEIGHT = 28

argv = sys.argv
argc = len(argv)

print('%s display a small portion of MNST' % argv[0])
print('[usage] python %s <label(.npy)> <image(.npy)>' % argv[0])

if argc < 3:
    quit()

data1 = np.load(argv[1])
data2 = np.load(argv[2])

if len(data1.shape) == 1 and len(data2.shape) == 3:
    labels = data1
    images = data2
elif len(data1.shape) == 3 and len(data2.shape) == 1:
    labels = data2
    images = data1
else:
    print('unexpected data')
    print('%s:' % argv[1], data1.shape)
    print('%s:' % argv[2], data2.shape)
    quit()

row_imgs = np.zeros((0, WIDTH * NR_IMAGES), np.uint8)

for i in range(10):

    indices = np.where(labels == i)[0]
    
    col_imgs = np.zeros((HEIGHT, 0), np.uint8)
    for index in indices[:NR_IMAGES]:
        img = images[index]
        col_imgs = np.hstack((col_imgs, img))

    row_imgs = np.vstack((row_imgs, col_imgs))

cv2.imshow('partial', row_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('partial.png', row_imgs)
print('save partial.png')
