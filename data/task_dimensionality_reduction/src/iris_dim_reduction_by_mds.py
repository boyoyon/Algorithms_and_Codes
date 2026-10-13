import numpy as np
import matplotlib.pylab as plt
from sklearn import manifold

features = np.load('..\\data\\iris_features.npy')
label = np.load('..\\data\iris_label.npy')

indices0 = np.where(label == 0)[0]
indices1 = np.where(label == 1)[0]
indices2 = np.where(label == 2)[0]

N = features.shape[0]
P = np.empty((N, N), np.float32)

for y in range(N):
    for x in range(N):
        P[y][x] = np.linalg.norm(features[y]-features[x])

P2 = P ** 2 # 各要素を2乗

# MDSオブジェクトの作成
mds = manifold.MDS(n_components = 2,  dissimilarity = 'precomputed')

# MDSの実行
Xd = mds.fit_transform(P2)

plt.scatter(Xd[indices0,0], Xd[indices0,1], color='red', label='setosa')
plt.scatter(Xd[indices1,0], Xd[indices1,1], color='green', label='versicolor')
plt.scatter(Xd[indices2,0], Xd[indices2,1], color='orange', label='virginica')
plt.legend()
plt.tight_layout()
plt.show()