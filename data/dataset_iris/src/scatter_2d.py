import numpy as np
import matplotlib.pyplot as plt

features = np.load('..\\data\\features.npy')
label = np.load('..\\data\\label_encoded.npy')

indices0 = np.where(label == 0)[0]
indices1 = np.where(label == 1)[0]
indices2 = np.where(label == 2)[0]

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# [0][0] sepal.length vs sepal.width

x0 = features[indices0, 0]
y0 = features[indices0, 1]
axes[0][0].scatter(x0, y0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 1]
axes[0][0].scatter(x1, y1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 1]
axes[0][0].scatter(x2, y2, color='orange')

axes[0][0].set_title('sepal.length vs sepal.width')

# [0][1] sepal.length vs petal.length

x0 = features[indices0, 0]
y0 = features[indices0, 2]
axes[0][1].scatter(x0, y0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 2]
axes[0][1].scatter(x1, y1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 2]
axes[0][1].scatter(x2, y2, color='orange')

axes[0][1].set_title('sepal.length vs petal.length')


# [0][2] sepal.length vs petal.width

x0 = features[indices0, 0]
y0 = features[indices0, 3]
axes[0][2].scatter(x0, y0, color='red')

x1 = features[indices1, 0]
y1 = features[indices1, 3]
axes[0][2].scatter(x1, y1, color='green')

x2 = features[indices2, 0]
y2 = features[indices2, 3]
axes[0][2].scatter(x2, y2, color='orange')

axes[0][2].set_title('sepal.length vs petal.width')

# [1][0] sepal.width vs petal.length

x0 = features[indices0, 1]
y0 = features[indices0, 2]
axes[1][0].scatter(x0, y0, color='red')

x1 = features[indices1, 1]
y1 = features[indices1, 2]
axes[1][0].scatter(x1, y1, color='green')

x2 = features[indices2, 1]
y2 = features[indices2, 2]
axes[1][0].scatter(x2, y2, color='orange')

axes[1][0].set_title('sepal.width vs petal.length')

# [1][1] sepal.width vs petal.width

x0 = features[indices0, 1]
y0 = features[indices0, 3]
axes[1][1].scatter(x0, y0, color='red')

x1 = features[indices1, 1]
y1 = features[indices1, 3]
axes[1][1].scatter(x1, y1, color='green')

x2 = features[indices2, 1]
y2 = features[indices2, 3]
axes[1][1].scatter(x2, y2, color='orange')

axes[1][1].set_title('sepal.width vs petal.width')

# [1][2] petal.length vs petal.width

x0 = features[indices0, 2]
y0 = features[indices0, 3]
axes[1][2].scatter(x0, y0, color='red')

x1 = features[indices1, 2]
y1 = features[indices1, 3]
axes[1][2].scatter(x1, y1, color='green')

x2 = features[indices2, 2]
y2 = features[indices2, 3]
axes[1][2].scatter(x2, y2, color='orange')

axes[1][2].set_title('petal.length vs petal.width')

plt.tight_layout()
plt.show()




