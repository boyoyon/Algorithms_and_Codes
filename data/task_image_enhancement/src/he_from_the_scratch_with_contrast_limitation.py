import numpy as np
import cv2, os, sys

argv= sys.argv
argc = len(argv)

# 1つのビンの度数が全ピクセルの2%を超えないように制限する
CLIP=2.0

if argc < 2:
    print('%s equalizes histogram with contrast limitation' % argv[0])
    print('%s <image> [<clip(2.0)>]' % argv[0])
    quit()

src = cv2.imread(argv[1])
height, width, chs = src.shape[:3]
nrPixels = height * width

if argc > 2:
    CLIP = float(argv[2])

th = int(CLIP * nrPixels / 100)

bgr = cv2.split(src)

for ch in range(chs):

    histogram = np.zeros((256), np.int32)
    newValue = np.zeros((256), np.int32)

    for y in range(height):

        for x in range(width):

            idx = bgr[ch][y][x]

            histogram[idx] += 1

    sum = 0
    clipped = 0

    for idx in range(256):
        if histogram[idx] > th:
            clipped += histogram[idx] - th
            histogram[idx] = th
        
    for idx in range(256):
        sum += histogram[idx] + clipped // 256
        newValue[idx] = 255 * sum // nrPixels

    for y in range(height):
        for x in range(width):
            idx = bgr[ch][y][x]
            bgr[ch][y][x] = newValue[idx]
    
dst = cv2.merge(bgr)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

print('Hit s-key to save and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)

if key == ord('s') or key == ord('S'):

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    clip = '%.2f' % CLIP
    clip = clip.replace('.', '_')

    dst_path = '%s_he_clipped_%s.png' % (filename, clip)
    cv2.imwrite(dst_path, dst)
    print('save %s' % dst_path)