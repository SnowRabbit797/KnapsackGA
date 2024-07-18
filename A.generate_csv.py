import csv
import numpy as np
import scipy.stats as st

# 正規分布で価値を生成 (負の値を再生成)
def generate_positive_normal_values(mean, std, size):
    values = st.norm.rvs(loc=mean, scale=std, size=size).astype(int)
    while any(values <= 0):
        values = np.where(values <= 0, st.norm.rvs(loc=mean, scale=std, size=size).astype(int), values)
    return values

# 一様分布で重さを生成 (負の値を再生成)
def generate_positive_uniform_values(low, high, size):
    values = st.uniform.rvs(loc=low, scale=high-low, size=size).astype(int)
    while any(values <= 0):
        values = np.where(values <= 0, st.uniform.rvs(loc=low, scale=high-low, size=size).astype(int), values)
    return values

# アイテム数
num_items = 500

# 価値の正規分布パラメータ
value_mean = 100
value_std = 50

# 重さの一様分布パラメータ
weight_min = 1
weight_max = 100

# 正の値を持つ価値と重さを生成
values = generate_positive_normal_values(value_mean, value_std, num_items)
weights = generate_positive_uniform_values(weight_min, weight_max, num_items)

# CSVファイルに書き込む
filename = 'items.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['value', 'weight'])
    for v, w in zip(values, weights):
        writer.writerow([v, w])
