import numpy as np
import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s detects line in the image by PHT' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

# 画像読み込み → グレースケール化
img = cv2.imread(argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# エッジ検出
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 確率的ハフ変換（PHT）による直線検出
# パラメータ: (画像, 距離解像度, 角度解像度, 閾値, 最小線長, 最大線隙間)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 検出した直線を元の画像に描画
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 赤色（BGR: 0, 0, 255）で線を描画
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('PHT', img)

print('Hit s-key to save result image')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):
    cv2.imwrite('result_PHT.png', img)

cv2.destroyAllWindows()
