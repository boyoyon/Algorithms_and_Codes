import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

features = np.load('..\\data\\iris_features.npy')
label = np.load('..\\data\iris_label.npy')

indices0 = np.where(label == 0)[0]
indices1 = np.where(label == 1)[0]
indices2 = np.where(label == 2)[0]

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.5)
Xd = kpca.fit_transform(features)

plt.scatter(Xd[indices0,0], Xd[indices0,1], color='red', label='setosa')
plt.scatter(Xd[indices1,0], Xd[indices1,1], color='green', label='versicolor')
plt.scatter(Xd[indices2,0], Xd[indices2,1], color='orange', label='virginica')
plt.legend()
plt.tight_layout()
plt.show()
