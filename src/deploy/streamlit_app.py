"""
Streamlit 前端：调用 Flask API 进行文本分类预测。

用法：
    1. 先启动 API：python src/deploy/api_flask_server.py --model bert
    2. 再启动前端：streamlit run src/deploy/streamlit_app.py
注意：前端选择的模型需与 API 启动时的 --model 参数一致。
"""
import os
import sys

# 将项目根目录加入 sys.path，保证任意位置运行时都能找到 config
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _CURRENT_DIR
while not os.path.exists(os.path.join(_PROJECT_ROOT, 'config.py')):
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import requests
import time
from config import Config

config = Config()

# 配置页面
st.title('中文新闻文本分类系统')

model_name = st.selectbox('选择模型（需与 API 启动参数 --model 一致）', ['rf', 'fasttext', 'bert', 'llm'])
text = st.text_input('请输入文本：')

# 定义接口地址 URL
url = f'http://{config.api_host}:{config.api_port}/predict'

if st.button('开始预测'):
    try:
        # 记录开始时间
        start_time = time.time()

        # 将文本封装成字典，向服务端发送 POST 请求
        data = {'text': text}
        result = requests.post(url, json=data)

        total_time = (time.time() - start_time) * 1000

        # 在页面上展示预测类别与推理时长
        st.success('预测的类别：' + result.json()['pred_class'] + ' | 推理的时长：' + str(round(total_time, 2)) + ' ms')

    except Exception as e:
        st.error('出问题了，请联系管理员')
        print('error:', e)
