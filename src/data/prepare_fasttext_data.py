"""
数据预处理：将原始文本转为 fasttext 需要的输入格式（txt 文件）。

fasttext 输入格式要求：每行 = "__label__类别名 分词结果（空格分隔）"
    e.g.  __label__finance 美 国 轰 炸 伊 朗      （按字分词）
          __label__finance 美国 轰炸 伊朗        （jieba 分词）
"""
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


def process_data(data_path, save_path, is_char=True):
    """
    :param data_path: 原始数据路径（tab 分隔：文本、标签索引）
    :param save_path: fasttext 格式数据保存路径
    :param is_char: True 按字分词，False 按 jieba 词分词
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        with open(save_path, 'w', encoding='utf-8') as fw:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 分割得到文本和标签索引
                text, label = line.split('\t')
                # 标签索引 -> 类别名
                label_name = config.id2class[int(label)]

                if is_char:
                    split_text = ' '.join(list(text))          # 按字分词
                else:
                    split_text = ' '.join(jieba.lcut(text))    # 按词分词

                fw.write('__label__' + label_name + ' ' + split_text + '\n')


if __name__ == '__main__':
    # 训练 / 验证 / 测试集，各生成按字、按词两种格式
    for src, dst, char in [
        (config.train_path, config.char_train_path, True),
        (config.train_path, config.word_train_path, False),
        (config.dev_path, config.char_dev_path, True),
        (config.dev_path, config.word_dev_path, False),
        (config.test_path, config.char_test_path, True),
        (config.test_path, config.word_test_path, False),
    ]:
        process_data(src, dst, is_char=char)
        print('已生成', dst)
