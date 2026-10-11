import gzip
import numpy as np

with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)

np.save('train_labels.npy', data)
print('save train_labels.npy')
print(data.shape)