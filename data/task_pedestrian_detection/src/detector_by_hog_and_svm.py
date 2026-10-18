import cv2, os, sys

argv = sys.argv
argc = len(argv)

print('%s detect pedestrians' % argv[0])
print('[usgae] python %s <input image>' % argv[0])

if argc < 2:
    quit()

# HOG検出器の初期化（人間検出用の学習済みモデルをロード）
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 画像の読み込み
image = cv2.imread(argv[1])

# 物体検出の実行
boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)

# 検出結果の描画
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('result', image)

print('Hit s-key to save result and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    # 画像保存
    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_hog_svm.png' % filename
    cv2.imwrite(dst_path, image)
    print('save %s' % dst_path)