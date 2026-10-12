import numpy as np
import pandas as pd

# iris.csv を読み込む。タイトル行は自動でスキップしてくれる
df = pd.read_csv('..\\data\\iris.csv')

# 特徴量（最初の4列）とラベル（最後の列）に分割する
features = df.iloc[:, :4].to_numpy()
label = df.iloc[:, -1].to_numpy()

print(features.shape)
print(label.shape)

# numpy形式 (.npy) で保存する
np.save("features.npy", features)
np.save("label.npy", label)

print("conversion completed: iris.csv --> features.npy， label.npy")
