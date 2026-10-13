import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

images = np.load('..\\data\\mnist_images20.npy')
labels = np.load('..\\data\mnist_labels20.npy')

cmap = plt.get_cmap('tab20')

pca = PCA(n_components = 2)
pca.fit(images)
Xd = pca.transform(images)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$") 

plt.legend()
plt.tight_layout()
plt.show()


