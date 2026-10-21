import cv2
import numpy as np

def compute_energy_map(img):
    """Sobelフィルタを用いて画像のエネルギーマップ（輪郭の強さ）を計算"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.abs(sobel_x) + np.abs(sobel_y)
    return energy_map

def find_vertical_seam(energy_map):
    """動的計画法(DP)で上から下までの最小エネルギー経路(シーム)を探索"""
    h, w = energy_map.shape
    dp = energy_map.copy()
    backtrack = np.zeros_like(dp, dtype=int)
    
    for i in range(1, h):
        for j in range(w):
            # 左上、真上、右上の3方向から最小の累積エネルギーを選択
            idx_min = j if j == 0 else j - 1 + np.argmin(dp[i-1, max(0, j-1):min(w, j+2)])
            if j > 0 and idx_min == j - 1:
                actual_idx = j - 1
            elif j < w - 1 and idx_min == j + 1:
                actual_idx = j + 1
            else:
                actual_idx = j
                
            dp[i, j] += dp[i-1, actual_idx]
            backtrack[i, j] = actual_idx

    # 最下層の最小値から上に向かってバックトラック
    seam = []
    j = np.argmin(dp[-1])
    seam.append(j)
    for i in range(h - 1, 0, -1):
        j = backtrack[i, j]
        seam.append(j)
        
    return seam[::-1]

def remove_vertical_seam(img, seam):
    """検出したシーム(1ピクセル幅)を画像から削除"""
    h, w, c = img.shape
    output = np.zeros((h, w - 1, c), dtype=img.dtype)
    for i in range(h):
        j = seam[i]
        # シームの左右を結合して1ピクセル縮小
        output[i, :, 0] = np.delete(img[i, :, 0], j)
        output[i, :, 1] = np.delete(img[i, :, 1], j)
        output[i, :, 2] = np.delete(img[i, :, 2], j)
    return output

# --- メイン処理 ---

import os, sys

scale_w = 0.8
scale_h = 1.0

argv = sys.argv
argc = len(argv)

print('%s executes seam carving from scratch' % argv[0])
print('[usage] python %s <image> [<scale_w(%.2f)> <scale_h(%.2f)>]' % (argv[0], scale_w, scale_h))

if argc < 2:
    quit()

if argc > 2:
    scale_w = float(argv[2])
    if scale_w > 1.0:
        print('scale_w exeeds 1.0. clip to 1.0')
        scale_w = 1.0

if argc > 3:
    scale_h = float(argv[3])
    if scale_h > 1.0:
        print('scale_h exeeds 1.0. clip to 1.0')
        scale_h = 1.0

print('%s, %.2f, %.2f' % (argv[1], scale_w, scale_h))

src = cv2.imread(argv[1])
cv2.imshow('source', src)
H, W = src.shape[:2]

h = int(H * scale_h)
w = int(W * scale_w)

dst = src.copy()

num_seams_to_remove = H - h
if num_seams_to_remove > 0:
    for i in range(num_seams_to_remove):
        print('vertical processing: %d/%d' % (i+1, num_seams_to_remove))
        energy = compute_energy_map(dst)
        seam = find_vertical_seam(energy)
        dst = remove_vertical_seam(dst, seam)

num_seams_to_remove = W - w
if num_seams_to_remove > 0:
    cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    for i in range(num_seams_to_remove):
        print('horizontal processing: %d/%d' % (i+1, num_seams_to_remove))
        energy = compute_energy_map(dst)
        seam = find_vertical_seam(energy)
        dst = remove_vertical_seam(dst, seam)
    cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('result', dst)

print('Hit s-key to save')
print('Hit any other key to quit')

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    str_scale_w = ('%.2f' % scale_w).replace('.', '_') 
    str_scale_h = ('%.2f' % scale_h).replace('.', '_') 
    dst_path = '%s_seam_carved_%s_%s.png' % (filename, str_scale_w, str_scale_h)
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)