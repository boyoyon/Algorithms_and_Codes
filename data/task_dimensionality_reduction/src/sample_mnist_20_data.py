import random
import numpy as np
import cv2

n = 20

images = np.load('test_images.npy')
labels = np.load('test_labels.npy')

rng = np.random.default_rng()

indices = []
for i in range(10):
    indices += random.sample(np.where(labels == i)[0].tolist(), n)

images20 = images[indices]
images20 = images20.reshape((20*10, 28*28))
np.save('mnist_images20.npy', images20)
print('save mnist_images20.npy')

labels20 = labels[indices]
np.save('mnist_labels20.npy', labels20)
print('save labels20.npy') 



