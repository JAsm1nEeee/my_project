"""
EDA（Exploratory Data Analysis）：探索性数据分析。

查看标签数量分布、句子长度分布，帮助发现数据规律或异常。
"""
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 将项目根目录加入 sys.path，保证任意位置运行时都能找到 config
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _CURRENT_DIR
while not os.path.exists(os.path.join(_PROJECT_ROOT, 'config.py')):
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from config import Config

config = Config()

# 1. 读取 txt 数据（tab 分隔），names 指定列名
train_data = pd.read_csv(config.train_path, sep='\t', names=['text', 'label'])
dev_data = pd.read_csv(config.dev_path, sep='\t', names=['text', 'label'])
test_data = pd.read_csv(config.test_path, sep='\t', names=['text', 'label'])
print('数据形状: train=', train_data.shape, ' dev=', dev_data.shape, ' test=', test_data.shape)
print('-' * 40)

# 2. 统计标签分布
label_count = Counter(train_data['label'])
print('标签分布:', label_count)

# 3. 统计句子长度（去空格后按字符数统计）
train_data['text_length'] = train_data['text'].str.replace(' ', '').str.len()
print(train_data['text_length'].describe())

# 4. 可视化：标签计数图 + 句子长度直方图
sns.countplot(x='label', data=train_data)
plt.show()

sns.histplot(train_data['text_length'], kde=False, bins=20)
plt.show()
