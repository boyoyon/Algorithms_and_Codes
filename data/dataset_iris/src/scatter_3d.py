import matplotlib.pyplot as plt
import numpy as np

features = np.load('..\\data\\features.npy')
label = np.load('..\\data\\label_encoded.npy')

indices0 = np.where(label == 0)[0]
indices1 = np.where(label == 1)[0]
indices2 = np.where(label == 2)[0]

fig, axes = plt.subplots(2, 2, figsize=(8, 6), subplot_kw={'projection': '3d'})

# [0][0] Excluding  petal.width

x0 = features[indices0, 0]
y0 = features[indices0, 1]
z0 = features[indices0, 2]
axes[0][0].scatter(x0, y0, z0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 1]
z1 = features[indices1, 2]
axes[0][0].scatter(x1, y1, z1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 1]
z2 = features[indices2, 2]
axes[0][0].scatter(x2, y2, z2, color='orange')

axes[0][0].set_title('Excluding petal.width')

# [0][1] Excluding  petal.length

x0 = features[indices0, 0]
y0 = features[indices0, 1]
z0 = features[indices0, 3]
axes[0][1].scatter(x0, y0, z0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 1]
z1 = features[indices1, 3]
axes[0][1].scatter(x1, y1, z1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 1]
z2 = features[indices2, 3]
axes[0][1].scatter(x2, y2, z2, color='orange')

axes[0][1].set_title('Excluding petal.length')

# [1][0] Excluding  sepal.width

x0 = features[indices0, 0]
y0 = features[indices0, 2]
z0 = features[indices0, 3]
axes[1][0].scatter(x0, y0, z0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 2]
z1 = features[indices1, 3]
axes[1][0].scatter(x1, y1, z1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 2]
z2 = features[indices2, 3]
axes[1][0].scatter(x2, y2, z2, color='orange')

axes[1][0].set_title('Excluding  sepal.width')

# [1][1] Excluding  sepal.length

x0 = features[indices0, 1]
y0 = features[indices0, 2]
z0 = features[indices0, 3]
axes[1][1].scatter(x0, y0, z0, color='red')

x1 = features[indices1, 1]
y1 = features[indices1, 2]
z1 = features[indices1, 3]
axes[1][1].scatter(x1, y1, z1, color='green')

x2 = features[indices2, 1]
y2 = features[indices2, 2]
z2 = features[indices2, 3]
axes[1][1].scatter(x2, y2, z2, color='orange')

axes[1][1].set_title('Excluding  sepal.length')

plt.tight_layout()
plt.show()
