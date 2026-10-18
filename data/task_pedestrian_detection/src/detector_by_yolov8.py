import cv2, os, sys
import numpy as np

argv = sys.argv
argc = len(argv)

print('%s detect pedestrians' % argv[0])
print('[usgae] python %s <input image>' % argv[0])

if argc < 2:
    quit()


# --- 1. 定数の設定 ---
# COCOデータセット（YOLOの既定モデルが学習している80クラスの名前）
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "over", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ランダムな枠線の色をクラスごとに生成
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# --- 2. モデルの読み込み ---
# 先ほど書き出したONNXファイルをOpenCV DNNにロード
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# --- 3. 画像の読み込みと前処理 ---
image = cv2.imread(argv[1])
if image is None:
    print("画像ファイルが見つかりません。")
    exit()

h, w, _ = image.shape

# YOLOv8に入力するために画像を640x640のBlob（数値配列）に変換
# 1/255.0でピクセル値を0〜1に正規化
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)

# --- 4. 予測（推論）の実行 ---
# ネットワークに画像を流して出力を得る
outputs = net.forward()

# YOLOv8の出力形状は [1, 84, 8400]。これを転置して扱いやすくする
# 84の内訳は：[x_center, y_center, width, height, クラス0の確率, クラス1の確率 ... ]
rows = outputs[0].T

boxes = []
confidences = []
class_ids = []

# 画像サイズ（640x640）に対する倍率を計算
x_factor = w / 640
y_factor = h / 640

# --- 5. 出力データの解析（ポストプロセス） ---
for row in rows:
    classes_scores = row[4:]
    max_score = np.amax(classes_scores)
    
    # 確信度が0.4以上のものだけを検出とする
    if max_score >= 0.4:
        class_id = np.argmax(classes_scores)
        
        # 矩形の中心座標とサイズを取得
        cx, cy, width, height = row[0], row[1], row[2], row[3]
        
        # 元の画像サイズに座標を変換（中心座標から左上座標に変換）
        left = int((cx - width / 2) * x_factor)
        top = int((cy - height / 2) * y_factor)
        width = int(width * x_factor)
        height = int(height * y_factor)
        
        boxes.append([left, top, width, height])
        confidences.append(float(max_score))
        class_ids.append(class_id)

# 重複するバウンディングボックスを排除（NMS処理）
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

# --- 6. 画面への描画 ---
# indices が空でないか確認し、1次元配列（あるいはリスト）としてループを回します
if len(indices) > 0:
    # 完全に1次元のリスト/配列にフラット化してループ
    for index in indices.flatten():
        box = boxes[index]
        left, top, width, height = box[0], box[1], box[2], box[3]
        
        label = f"{CLASSES[class_ids[index]]}: {confidences[index]:.2f}"
        color = COLORS[class_ids[index]]
        
        # 矩形を描画
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        # テキストを描画
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 結果の表示
cv2.imshow('YOLOv8 OpenCV DNN', image)

print('Hit s-key to save result and terminate')
print('Hit any other key to quit')

key = cv2.waitKey(0)
cv2.destroyAllWindows()

if key == ord('s') or key == ord('S'):

    # 画像保存
    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]
    dst_path = '%s_yolov8.png' % filename
    cv2.imwrite(dst_path, image)
    print('save %s' % dst_path)