import numpy as np
import matplotlib.pylab as plt
from sklearn import manifold

images = np.load('..\\data\\coil20_images.npy')
labels = np.load('..\\data\coil20_labels.npy')

cmap = plt.get_cmap('tab20')

N = images.shape[0]
P = np.zeros((N, N), np.float32)

for y in range(N):
    for x in range(N):
        if x < y:
            continue
        P[y][x] = P[x][y] = np.linalg.norm(images[y]-images[x])

P2 = P ** 2 # 各要素を2乗

# MDSオブジェクトの作成
mds = manifold.MDS(n_components = 2,  dissimilarity = 'precomputed')

# MDSの実行
Xd = mds.fit_transform(P2)

for i in range(20):
    indices = np.where(labels == i)[0]
    plt.scatter(Xd[indices,0], Xd[indices,1], color=cmap(i), marker=f"${i}$") 

plt.legend()
plt.tight_layout()
plt.show()