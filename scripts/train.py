"""
统一训练入口。

用法：
    python scripts/train.py --model rf
    python scripts/train.py --model fasttext --mode char --tune auto
    python scripts/train.py --model bert
"""
import argparse
import os
import runpy
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# 模型 -> 训练脚本路径 映射
TRAIN_SCRIPTS = {
    'rf': os.path.join(_PROJECT_ROOT, 'src', 'models', 'rf', 'rf_train.py'),
    'fasttext': os.path.join(_PROJECT_ROOT, 'src', 'models', 'fasttext', 'fasttext_train.py'),
    'bert': os.path.join(_PROJECT_ROOT, 'src', 'models', 'bert', 'bert_train.py'),
}


def main():
    parser = argparse.ArgumentParser(description='模型训练统一入口')
    parser.add_argument('--model', choices=list(TRAIN_SCRIPTS.keys()), required=True, help='选择要训练的模型')
    args, remain = parser.parse_known_args()

    script = TRAIN_SCRIPTS[args.model]
    # 将剩余参数（如 --mode / --tune）透传给具体训练脚本
    sys.argv = [script] + remain
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
