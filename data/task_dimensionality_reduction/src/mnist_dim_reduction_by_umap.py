# pip install umap-learn

import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_iris
import numpy as np

images = np.load('..\\data\\mnist_images20.npy')
labels = np.load('..\\data\mnist_labels20.npy')

cmap = plt.get_cmap('tab20')

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
Xd = reducer.fit_transform(images)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$") 

plt.legend()
plt.tight_layout()
plt.show()