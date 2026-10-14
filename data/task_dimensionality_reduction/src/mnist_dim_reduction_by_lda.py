import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

images = np.load('..\\data\\mnist_images20.npy')
labels = np.load('..\\data\mnist_labels20.npy')

cmap = plt.get_cmap('tab20')

lda = LinearDiscriminantAnalysis(n_components=2)
Xd = lda.fit_transform(images, labels)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$") 

plt.legend()
plt.tight_layout()
plt.show()