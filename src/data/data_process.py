"""
数据预处理：逐行 jieba 分词并保存为 CSV（随机森林模型使用）。

流程：
    1. 读取 tab 分隔的原始数据（text, label）
    2. 逐行 jieba 分词，序列长度截断为前 20 个词
    3. 统计分词序列长度分布
    4. 保存处理结果到 CSV（新增 words / seq_len 两列）
"""
import pandas as pd
import jieba
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


def process_data(data_path, save_path):
    """对单份数据集执行 jieba 分词并保存为 CSV"""
    # 1. 读取原始数据
    data = pd.read_csv(data_path, sep='\t', names=['text', 'label'])

    # 2. 逐行 jieba 分词，并限制切词后序列不超过 20 个词
    def cut_words(sentence):
        split_word = jieba.lcut(sentence)[:20]
        return ' '.join(split_word), len(split_word)

    data['words'], data['seq_len'] = zip(*data['text'].apply(cut_words))

    # 3. 查看分词后序列长度的分布
    print(data['seq_len'].describe())

    # 4. 保存处理结果
    data.to_csv(save_path, index=False)
    print('数据保存在', save_path)


if __name__ == '__main__':
    process_data(config.train_path, config.process_train_path)
    process_data(config.dev_path, config.process_dev_path)
    process_data(config.test_path, config.process_test_path)
