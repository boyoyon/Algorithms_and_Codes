# pip install seam-carving opencv-python

import cv2, os, sys
import seam_carving

scale_w = 0.8
scale_h = 1.0

argv = sys.argv
argc = len(argv)

print('%s executes seam carving by seam-carving library' % argv[0])
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

# 画像の読み込み
img = cv2.imread(argv[1])
cv2.imshow('sourface', img)

h, w, c = img.shape

# 変更後のサイズを指定 
new_w = int(w * scale_w)
new_h = int(h * scale_h)

# シームカービングの実行 (NumPy配列をそのまま渡せます)
# ※ backwardエネルギーのほか、より歪みの少ないforwardエネルギーも指定可能
output = seam_carving.resize(
    img, 
    size=(new_w, new_h), 
    energy_mode='backward',  # または 'forward'
    order='width-first'
)

# 結果の表示
cv2.imshow('result', output)

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
    cv2.imwrite(dst_path, output)
    print('save %s' % dst_path)
