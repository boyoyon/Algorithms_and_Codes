import numpy as np
import cv2

images = np.load('..\\data\\coil20_images.npy')

no = 0
row_image = np.zeros((0, 32*72), np.uint8)
for obj in range(20):

    col_image = np.zeros((32, 0), np.uint8)
    for angle in range(72):

        img = images[no]
        no += 1
        img = img.reshape((32,32))
        col_image = np.hstack((col_image, img))

    row_image = np.vstack((row_image, col_image))

cv2.imshow('images', row_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
         