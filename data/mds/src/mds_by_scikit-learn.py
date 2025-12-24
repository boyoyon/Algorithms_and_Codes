import numpy as np
import matplotlib.pylab as plt
from sklearn import manifold

# 近接行列(の各要素を2乗したもの)
P = np.array([
    [0.00, 1.00, 1.73, 2.00, 1.73, 1.00],
    [1.00, 0.00, 1.00, 1.73, 2.00, 1.73],
    [1.73, 1.00, 0.00, 1.00, 1.73, 2.00],
    [2.00, 1.73, 1.00, 0.00, 1.00, 1.73],
    [1.73, 2.00, 1.73, 1.00, 0.00, 1.00],
    [1.00, 1.73, 2.00, 1.73, 1.00, 0.00]])

P2 = P ** 2 # 各要素を2乗

N = P2.shape[0]

# MDSオブジェクトの作成
mds = manifold.MDS(n_components = 2,  dissimilarity = 'precomputed')

# MDSの実行
loc = mds.fit_transform(P2)

# MDS実行結果の表示
for i in range(N):
    plt.plot([loc[i-1][0], loc[i][0]], [loc[i-1][1], loc[i][1]])

plt.show()