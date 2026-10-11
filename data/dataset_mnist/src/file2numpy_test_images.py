import gzip
import numpy as np

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)

data = data.reshape((10000, 28,28))
np.save('test_images.npy', data)
print('save train_images.npy')
print(data.shape)