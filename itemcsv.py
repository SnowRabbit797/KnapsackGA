import csv
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
def load_items_from_csv(filename):
    values = []
    weights = []
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            values.append(int(row['value']))
            weights.append(int(row['weight']))
    return values, weights

# ファイル名
filename = 'items.csv'

# データの読み込み
values, weights = load_items_from_csv(filename)

# ヒストグラムの作成
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(values, bins=20, edgecolor='black')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(weights, bins=20, edgecolor='black')
plt.title('Histogram of Weights')
plt.xlabel('Weight')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 散布図の作成
plt.figure(figsize=(7, 6))
plt.scatter(values, weights, alpha=0.6)
plt.title('Scatter Plot of Values and Weights')
plt.xlabel('Value')
plt.ylabel('Weight')
plt.grid(True)
plt.show()
