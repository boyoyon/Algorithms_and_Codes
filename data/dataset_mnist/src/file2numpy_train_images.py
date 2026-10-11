import gzip
import numpy as np

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)

data = data.reshape((60000, 28,28))
np.save('train_images.npy', data)
print('save train_images.npy')
print(data.shape)