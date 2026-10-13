import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

features = np.load('..\\data\\iris_features.npy')
label = np.load('..\\data\iris_label.npy')

indices0 = np.where(label == 0)[0]
indices1 = np.where(label == 1)[0]
indices2 = np.where(label == 2)[0]

#lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)

lle = LocallyLinearEmbedding(n_neighbors=2, n_components=2, random_state=42)
Xd = lle.fit_transform(features)

plt.scatter(Xd[indices0,0], Xd[indices0,1], color='red', label='setosa')
plt.scatter(Xd[indices1,0], Xd[indices1,1], color='green', label='versicolor')
plt.scatter(Xd[indices2,0], Xd[indices2,1], color='orange', label='virginica')
plt.legend()
plt.tight_layout()
plt.show()