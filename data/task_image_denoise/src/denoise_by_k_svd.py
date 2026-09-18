import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage import data, img_as_float
from skimage.util import random_noise

import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s denoises the image by k-SVD' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

src = cv2.imread(argv[1])
cv2.imshow('noisy', src)

noisy = src.copy() 
noisy = noisy.astype(np.float32) / 255.0

# 画像パッチの抽出
# K-SVDアプローチと同様に、画像を小さなオーバーラップパッチに分解する
patch_size = (8, 8)
patches = extract_patches_2d(noisy, patch_size)

# パッチを (枚数, 64) の2次元配列に平坦化（フラット化）
n_patches = patches.shape[0]
X = patches.reshape(n_patches, -1)

# 平均を引いて中心化（辞書学習の一般的な前処理）
intercept = np.mean(X, axis=1, keepdims=True)
X -= intercept

# 辞書学習とスパースコーディングの実行
# scikit-imageのMiniBatchDictionaryLearningを利用（K-SVDと類似の処理を高速に行う）
# n_components: 辞書の原子（要素）数。ここでは過完全（Overcomplete）辞書を設定
# transform_alpha: スパース性の制約強度のパラメータ
print("辞書学習を実行中...")
dico = MiniBatchDictionaryLearning(n_components=128, alpha=1.0, transform_alpha=1.0,
                                  max_iter=15, random_state=42, n_jobs=-1)
code = dico.fit_transform(X)

# パッチの再構成（ノイズ除去）
# スパースな係数（code）と学習された辞書（components_）を掛け合わせてパッチを復元
X_reconstructed = np.dot(code, dico.components_)

# 前処理で引いた平均を元に戻す
X_reconstructed += intercept

# 平坦なパッチを元の (枚数, 8, 8) の形状に戻す
patches_reconstructed = X_reconstructed.reshape(patches.shape)

# オーバーラップしたパッチを1枚の画像に再結合
# 重複する画素は平均化され、これによってノイズが平滑化される
denoised = reconstruct_from_patches_2d(patches_reconstructed, src.shape)

denoised *= 255.0
denozed = np.clip(denoised, 0, 255.0)
denoised = denoised.astype(np.uint8)

cv2.imshow('denoised', denoised)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_denosed_by_k_svd.png' % filename
    cv2.imwrite(dst_path, denoised)
    print('save %s' % dst_path)

cv2.destroyAllWindows()

