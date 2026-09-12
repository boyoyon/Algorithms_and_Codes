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

# Edge Drawing (ED) オブジェクトの作成
ed = cv2.ximgproc.createEdgeDrawing()

# パラメータのカスタマイズ
# 構造体を新しく作らず、ed.params の値を直接書き換えます
ed.Params.EdgeDetectionOperator = cv2.ximgproc.EdgeDrawing_PREWITT  # 勾配算子 (デフォルトは PREWITT)
ed.Params.GradientThresholdValue = 20    # アンカーポイント選定の閾値
ed.Params.EdgeArgument = 11                 # 勾配の閾値
ed.Params.MinPathLength = 10                # 最低限必要なエッジの長さ

# エッジと直線の検出
ed.detectEdges(gray)
lines = ed.detectLines()

# 検出された直線を元の画像に描画
# detectLines() の戻り値は [[[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ...] という形状の配列
output_img = img.copy()

if lines is not None:
    for line in lines:
        # 座標データを取得
        x1, y1, x2, y2 = line[0].astype(int)
        
        # 赤色（BGR: 0, 0, 255）で線を描画
        cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
else:
    print("直線が検出されませんでした。")
    quit()

cv2.imshow('EDLines', output_img)

print('Hit s-key to save result image')
print('Hit any other key to quit')
key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):
    cv2.imwrite('result_EDLines.png', output_img)

cv2.destroyAllWindows()
