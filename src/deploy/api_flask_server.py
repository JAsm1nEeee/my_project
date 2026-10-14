"""
Flask API 服务：统一入口，通过 --model 参数选择要部署的模型。

用法：
    python src/deploy/api_flask_server.py --model rf
    python src/deploy/api_flask_server.py --model fasttext
    python src/deploy/api_flask_server.py --model bert
    python src/deploy/api_flask_server.py --model llm

接口：POST http://127.0.0.1:6000/predict
    Body: {"text": "文本内容"}
    返回: {"text": ..., "pred_class": 预测类别名称}
"""
import argparse
import importlib
import os
import sys

# 将项目根目录加入 sys.path，保证任意位置运行时都能找到 config
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _CURRENT_DIR
while not os.path.exists(os.path.join(_PROJECT_ROOT, 'config.py')):
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, jsonify
from config import Config

config = Config()
app = Flask(__name__)

# 模型名 -> 预测函数模块映射
PREDICT_MODULES = {
    'rf': 'src.models.rf.rf_predict_fun',
    'fasttext': 'src.models.fasttext.fasttext_predict_fun',
    'bert': 'src.models.bert.bert_predict_fun',
    'llm': 'src.models.llm.deepseek_predict_fun',
}


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口：接收 JSON 数据，调用模型预测函数并返回结果"""
    data = request.get_json()
    result = app.config['PREDICT_FUN'](data)
    return jsonify(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型 API 服务')
    parser.add_argument('--model', choices=list(PREDICT_MODULES.keys()), default='rf', help='选择要部署的模型')
    args = parser.parse_args()

    # 按需加载对应模型的预测函数
    predict_fun = importlib.import_module(PREDICT_MODULES[args.model]).predict_fun
    app.config['PREDICT_FUN'] = predict_fun

    print(f'模型 [{args.model}] 的 API 服务已启动: http://{config.api_host}:{config.api_port}/predict')
    app.run(host=config.api_host, port=config.api_port, debug=True)
