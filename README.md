# 中文新闻文本自动分类系统

基于「传统机器学习 → 词嵌入模型 → 预训练模型 → 大模型」完整对比链路的中文短文本多分类项目。

输入一条中文新闻标题（短文本），自动将其归入 **10 个类别** 之一：

finance（财经）、realty（房产）、stocks（股票）、education（教育）、science（科技）、society（社会）、politics（时政）、sports（体育）、game（游戏）、entertainment（娱乐）。

## 项目结构

```
news_classification_project/
├── config.py                 # 统一配置（数据/模型/超参/API，路径基于项目根自动计算）
├── data/
│   ├── class.txt             # 类别列表（10 类）
│   ├── stopwords.txt         # 停用词表
│   ├── raw/                  # 原始数据（含重复样本，不入库）
│   ├── processed/            # 去重后数据 + jieba 分词 CSV（不入库）
│   └── fasttext/             # fasttext 输入格式（按字/按词，不入库）
├── models/
│   ├── pretrained/           # 预训练 bert-base-chinese（不入库）
│   └── saved/                # 各模型训练产物（不入库）
├── src/
│   ├── data/                 # 数据预处理与 EDA
│   ├── models/
│   │   ├── rf/               # 随机森林（TF-IDF 特征）
│   │   ├── fasttext/         # fasttext 分类（字符级/词级 × 默认/自动调参）
│   │   ├── bert/             # BERT 微调 + 动态量化
│   │   └── llm/              # DeepSeek 提示词零样本分类
│   └── deploy/               # Flask API + Streamlit 前端
├── scripts/
│   ├── prepare_data.py       # 一键数据准备（去重 → 分词 → fasttext 格式）
│   ├── train.py              # 统一训练入口
│   └── predict.py            # 统一推理入口
├── requirements.txt          # 依赖清单
└── .env                      # DeepSeek API Key（不入库，参考 .env.example）
```

## 环境

- Python 3.10+（推荐 3.10/3.11）
- 主要依赖：`pandas`、`jieba`、`scikit-learn`、`fasttext`、`torch`、`transformers`、`flask`、`streamlit`、`requests`、`openai`、`python-dotenv`、`matplotlib`、`seaborn`
- `requirements.txt` 依赖清单见仓库根目录（可自行用 `pip freeze` 生成）
- LLM 方案需要 `DeepSeek API Key`：复制 `.env.example` 为 `.env` 并填入

## 数据

- 10 类均衡文本数据，训练集 18 万 / 验证集 1 万 / 测试集 1 万
- 数据与模型体积较大，未纳入 Git 仓库；本地放置于 `data/` 与 `models/` 目录即可运行
- 重新生成预处理数据：`python scripts/prepare_data.py`（去重 → jieba 分词 → fasttext 格式）

## 训练

```bash
# 随机森林基线（TF-IDF + RandomForest）
python scripts/train.py --model rf

# fasttext（四种组合：字符级/词级 × 默认/自动调参）
python scripts/train.py --model fasttext --mode char --tune default
python scripts/train.py --model fasttext --mode word  --tune auto

# BERT 微调（bert-base-chinese）
python scripts/train.py --model bert

# BERT 动态量化（CPU 上对 Linear 层做 int8 量化，模型体积约缩小 2.6 倍）
python src/models/bert/bert_quantization.py
```

## 评估

- 随机森林：`python src/models/rf/rf_test.py`（在验证集上输出 accuracy / macro-F1 等）
- fasttext：训练脚本会自动在测试集上输出 P / R / F1
- BERT：训练脚本按验证集 F1 保存最优模型，量化脚本对比量化前后指标

| 方案 | Accuracy | macro-F1 |
|---|---|---|
| RandomForest（TF-IDF） | 0.733 | 0.740 |
| fasttext（词级 + auto） | 0.920 | 0.920 |
| BERT 微调（最终方案） | **0.9422** | **0.9421** |
| BERT 动态量化 | — | 0.9325 |

## 推理

```bash
# 单条文本预测
python scripts/predict.py --model rf --text "午评沪指涨0.78%逼近2800"
python scripts/predict.py --model bert --text "午评沪指涨0.78%逼近2800"
python scripts/predict.py --model llm --text "午评沪指涨0.78%逼近2800"
```

## 部署

```bash
# 1. 启动 Flask API（--model 可选 rf / fasttext / bert / llm）
python src/deploy/api_flask_server.py --model bert

# 2. 另开终端启动 Streamlit 前端
streamlit run src/deploy/streamlit_app.py
```

- API：`POST http://127.0.0.1:6000/predict`，Body：`{"text": "文本内容"}`，返回 `{"text": ..., "pred_class": ...}`
- 前端：浏览器打开 Streamlit 页面，选择与 API 一致的模型后输入文本即可预测
