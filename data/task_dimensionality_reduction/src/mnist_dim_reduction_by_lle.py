import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

images = np.load('..\\data\\mnist_images20.npy')
labels = np.load('..\\data\mnist_labels20.npy')

cmap = plt.get_cmap('tab20')

#lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)

lle = LocallyLinearEmbedding(n_neighbors=2, n_components=2, random_state=42)

Xd = lle.fit_transform(images)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$") 

plt.legend()
plt.tight_layout()
plt.show()