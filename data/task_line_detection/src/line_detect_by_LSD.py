import numpy as np
import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s detects line in the image by LSD' % argv[0])
print('[usage] python %s <image>' % argv[0])

if argc < 2:
    quit()

# 画像読み込み → グレースケール化
img = cv2.imread (argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# LSDオブジェクト作成～線分検出
lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(gray)[0]

# 検出した線分を元の画像に描画
for l in lines:
    x0 = int(l[0][0])
    y0 = int(l[0][1])
    x1 = int(l[0][2])
    y1 = int(l[0][3])
    
    # 赤色（BGR: 0, 0, 255）で線を描画
    cv2.line(img, (x0, y0), (x1, y1), ( 0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('LSD', img)

print('Hit s-key to save result image')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):
    cv2.imwrite('result_LSD.png', img)

cv2.destroyAllWindows()
