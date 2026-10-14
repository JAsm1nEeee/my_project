"""
统一推理入口。

用法：
    python scripts/predict.py --model rf --text "午评沪指涨0.78%逼近2800"
    python scripts/predict.py --model fasttext --text "午评沪指涨0.78%逼近2800"
    python scripts/predict.py --model bert --text "午评沪指涨0.78%逼近2800"
    python scripts/predict.py --model llm --text "午评沪指涨0.78%逼近2800"
"""
import argparse
import importlib
import os
import sys
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# 模型 -> 预测函数模块映射
PREDICT_MODULES = {
    'rf': 'src.models.rf.rf_predict_fun',
    'fasttext': 'src.models.fasttext.fasttext_predict_fun',
    'bert': 'src.models.bert.bert_predict_fun',
    'llm': 'src.models.llm.deepseek_predict_fun',
}


def main():
    parser = argparse.ArgumentParser(description='模型推理统一入口')
    parser.add_argument('--model', choices=list(PREDICT_MODULES.keys()), required=True, help='选择模型')
    parser.add_argument('--text', required=True, help='待分类的文本')
    args = parser.parse_args()

    # 按需加载对应模型的预测函数（避免一次性加载所有模型占用内存）
    predict_fun = importlib.import_module(PREDICT_MODULES[args.model]).predict_fun

    data = {'text': args.text}
    start = time.time()
    result = predict_fun(data)
    cost = (time.time() - start) * 1000

    print('文本:', result['text'])
    print('预测类别:', result['pred_class'])
    print('耗时: %.2f ms' % cost)


if __name__ == '__main__':
    main()
