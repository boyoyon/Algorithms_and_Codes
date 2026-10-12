import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# iris.csv を読み込む。タイトル行は自動でスキップしてくれる
df = pd.read_csv('..\\data\\iris.csv')

# 特徴量（最初の4列）とラベル（最後の列）に分割する
features = df.iloc[:, :4].to_numpy()

le = LabelEncoder()
label_encoded = le.fit_transform(df.iloc[:, -1])

print(features.shape)
print(label_encoded.shape)

# numpy形式 (.npy) で保存する
np.save("features.npy", features)
np.save("label_encoded.npy", label_encoded)

print("conversion completed: iris.csv --> features.npy， label_encoded.npy")
