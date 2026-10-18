# https://github.com/chuanqi305/MobileNet-SSD/blob/master/voc/MobileNetSSD_deploy.prototxt

# https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel

import cv2, os, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s detect pedestrians' % argv[0])
print('[usgae] python %s <input image>' % argv[0])

if argc < 2:
    quit()

# 1. モデルと設定ファイルの読み込み
prototxt_path = os.path.join(os.path.dirname(__file__), 'MobileNetSSD_deploy.prototxt')
model_path = os.path.join(os.path.dirname(__file__), 'MobileNetSSD_deploy.caffemodel')
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# MobileNet-SSDが学習している21個のクラス定義
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 2. テスト画像の読み込み
image = cv2.imread(argv[1])
(h, w) = image.shape[:2]

# 3. 画像をSSDの入力サイズ（300x300、平均値差し引き）に変換してモデルにセット
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)

# 4. 推論の実行
detections = net.forward()

# 5. 検出結果の描画
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # 確信度が50%以上のものだけを描画
    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])
        
        # 物体の座標を計算
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # 枠とラベルを描画
        label = f"{CLASSES[class_id]}: {confidence * 100:.1f}%"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 6. 結果の表示
cv2.imshow("Output", image)

print('Hit s-key to save result and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    # 画像保存
    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_ssd.png' % filename
    cv2.imwrite(dst_path, image)
    print('save %s' % dst_path)
