import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

images = np.load('..\\data\\coil20_images.npy')
labels = np.load('..\\data\coil20_labels.npy')

cmap = plt.get_cmap('tab20')

model = TSNE(n_components=2, perplexity=50, max_iter=500, verbose=3, random_state=1)

Xd = model.fit_transform(images)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$", label='obj%d' % (i+1)) 

#plt.legend()
plt.tight_layout()
plt.show()


