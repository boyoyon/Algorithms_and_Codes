import numpy as np
import matplotlib.pylab as plt

# STEP1: 近接行列の各要素を2乗したものを用意する

P = np.array([[0.00, 1.00, 1.73, 2.00, 1.73, 1.00],
    [1.00, 0.00, 1.00, 1.73, 2.00, 1.73],
    [1.73, 1.00, 0.00, 1.00, 1.73, 2.00],
    [2.00, 1.73, 1.00, 0.00, 1.00, 1.73],
    [1.73, 2.00, 1.73, 1.00, 0.00, 1.00],
    [1.00, 1.73, 2.00, 1.73, 1.00, 0.00]])

P2 = P ** 2 # 各要素を2乗

N = P2.shape[0]

# STEP2: センタリング行列を使ってダブルセンタリングを実施する。

C = np.eye(N) - np.ones((N,N))/2 # センタリング行列

B = - 1/2 * C * P2 * C # ヤング・ハウスホルダー変換 (ダブルセンタリング)

# STEP3: 固有値分解

w,v = np.linalg.eig(B)
indices = np.argsort(w)
x = indices[-1] # 1番大きい固有値の index をx軸とする
y = indices[-2] # 2番目に大きい固有値のindx を y軸とする

# MDS実行結果の表示
for i in range(N):
    plt.plot([v[i-1][x], v[i][x]], [v[i-1][y], v[i][y]])

plt.show()