import numpy as np
import cv2, glob, os

NR_OBJS = 20
NR_IMAGES = 72
WIDTH = 32
HEIGHT = 32

images = np.zeros((NR_OBJS * NR_IMAGES, WIDTH * HEIGHT), np.uint8)
labels = []
idx = 0
for obj in range(1,20+1):
    for no in range(72):
        path_name = os.path.join('coil-20-proc', 'obj%d__%d.png' % (obj, no))
        print(path_name)
        img = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        images[idx] = img.reshape((WIDTH*HEIGHT))
        idx += 1

        labels.append(obj)

np.save('coil20_images.npy', images)
print('save coil20_images.npy')

np.save('coil20_labels.npy', labels)
print('save coil20_labels.npy')