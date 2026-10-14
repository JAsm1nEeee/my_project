"""
一键数据准备：去重 -> jieba 分词（RF 用 CSV）-> fasttext 输入格式（char / word）。

用法：python scripts/prepare_data.py
"""
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from config import Config, remove_duplicates
from src.data.data_process import process_data as rf_process_data
from src.data.prepare_fasttext_data import process_data as ft_process_data

config = Config()


def main():
    print('1) 数据去重...')
    remove_duplicates(config.train_raw_path, config.train_path)
    remove_duplicates(config.dev_raw_path, config.dev_path)
    remove_duplicates(config.test_raw_path, config.test_path)

    print('2) jieba 分词 -> process_*.csv（随机森林使用）...')
    rf_process_data(config.train_path, config.process_train_path)
    rf_process_data(config.dev_path, config.process_dev_path)
    rf_process_data(config.test_path, config.process_test_path)

    print('3) 构建 fasttext 输入格式（char / word）...')
    for src, dst, is_char in [
        (config.train_path, config.char_train_path, True),
        (config.train_path, config.word_train_path, False),
        (config.dev_path, config.char_dev_path, True),
        (config.dev_path, config.word_dev_path, False),
        (config.test_path, config.char_test_path, True),
        (config.test_path, config.word_test_path, False),
    ]:
        ft_process_data(src, dst, is_char=is_char)
        print('   已生成', dst)

    print('数据准备全部完成。')


if __name__ == '__main__':
    main()
