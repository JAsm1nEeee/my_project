"""
统一配置文件：集中管理全局配置信息（数据路径、类别、模型路径、超参数、API 服务等）。

所有路径均基于项目根目录动态计算，项目整体移动 / 克隆到其他位置后无需修改即可运行。
"""
import os
from pathlib import Path  # 路径操作模块


class Config:
    """集中管理全局配置信息"""

    def __init__(self):
        # 1. 项目根目录：config.py 所在目录
        self.root_path = str(Path(__file__).resolve().parent)

        # ============ 2. 数据路径 ============
        self.data_path = os.path.join(self.root_path, "data")
        self.raw_path = os.path.join(self.data_path, "raw")
        self.processed_path = os.path.join(self.data_path, "processed")
        self.fasttext_path = os.path.join(self.data_path, "fasttext")

        # 原始数据（含少量完全一致的重复样本）
        self.train_raw_path = os.path.join(self.raw_path, "train_raw.txt")
        self.dev_raw_path = os.path.join(self.raw_path, "dev_raw.txt")
        self.test_raw_path = os.path.join(self.raw_path, "test_raw.txt")

        # 去重后数据集路径
        self.train_path = os.path.join(self.processed_path, "train.txt")
        self.dev_path = os.path.join(self.processed_path, "dev.txt")
        self.test_path = os.path.join(self.processed_path, "test.txt")

        # 停用词 / 类别文件路径
        self.stopwords_path = os.path.join(self.data_path, "stopwords.txt")
        self.class_path = os.path.join(self.data_path, "class.txt")

        # jieba 分词后的数据（随机森林模型使用）
        self.process_train_path = os.path.join(self.processed_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.processed_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.processed_path, "process_test.csv")

        # fasttext 需要的输入格式（按字 / 按词 两种）
        self.char_train_path = os.path.join(self.fasttext_path, "char_train.txt")
        self.char_dev_path = os.path.join(self.fasttext_path, "char_dev.txt")
        self.char_test_path = os.path.join(self.fasttext_path, "char_test.txt")
        self.word_train_path = os.path.join(self.fasttext_path, "word_train.txt")
        self.word_dev_path = os.path.join(self.fasttext_path, "word_dev.txt")
        self.word_test_path = os.path.join(self.fasttext_path, "word_test.txt")

        # ============ 3. 模型路径 ============
        self.models_path = os.path.join(self.root_path, "models")
        self.save_model_path = os.path.join(self.models_path, "saved")
        os.makedirs(self.save_model_path, exist_ok=True)  # 训练产物目录不存在则自动创建

        # 随机森林 + TF-IDF
        self.rf_model_path = os.path.join(self.save_model_path, "rf_model.pkl")
        self.tfidf_model_path = os.path.join(self.save_model_path, "tfidf_model.pkl")

        # fasttext（实际保存为 ft_model_char_1.bin / ft_model_char_2.bin / ft_model_word_1.bin / ft_model_word_2.bin）
        self.ft_model_path = os.path.join(self.save_model_path, "ft_model.bin")

        # BERT 微调 / 量化
        self.bert_model_path = os.path.join(self.save_model_path, "bert_model.pth")
        self.bert_quuantizaiton_model_path = os.path.join(self.save_model_path, "bert_quantization.pth")

        # 预训练 BERT：优先加载本地模型；本地不存在时回退到 HuggingFace 在线下载（模型名 bert-base-chinese）
        self.pretrained_path = os.path.join(self.models_path, "pretrained")
        self.bert_path = os.path.join(self.pretrained_path, "bert-base-chinese")
        if not os.path.exists(self.bert_path):
            self.bert_path = "bert-base-chinese"

        # 类别索引 -> 类别名称 的映射表（fasttext / bert 预测时使用）
        with open(self.class_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        self.id2class = {i: name for i, name in enumerate(class_list)}

        # ============ 4. BERT 微调超参数 ============
        # torch 为 BERT 相关脚本的依赖；rf / fasttext / llm 等模块无需安装 torch 也能运行
        try:
            import torch
            # 量化模型只能在 CPU 上运行，统一使用 CPU
            self.device = torch.device('cpu')
        except ImportError:
            self.device = None
        self.epochs = 5          # 训练轮数
        self.batch_size = 16     # 批次大小
        self.lr = 2e-5           # 微调学习率
        self.max_len = 32        # 最大长度（根据 EDA 的文本长度分布确定）
        self.weigth_decay = 1e-2 # AdamW 权重衰减系数

        # ============ 5. API 服务配置 ============
        self.api_host = '127.0.0.1'  # 部署在本机，仅允许本机访问
        self.api_port = 6000          # 端口号


# 对数据文件去重并保存到新文件
def remove_duplicates(input_file, output_file):
    """
    读取文件去重并保存。
    """
    seen = set()      # 集合：无序、自带去重、查询快
    total_lines = 0   # 原样本总数
    unique_lines = 0  # 去重后样本数

    # 确保输出目录存在，避免保存报错
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total_lines += 1
            if line not in seen:
                outfile.write(line)
                seen.add(line)
                unique_lines += 1

    print(f"去重前总行数: {total_lines}")
    print(f"去重后保留行数: {unique_lines}")
    print(f"重复行数: {total_lines - unique_lines}")
    print("-" * 30)


if __name__ == "__main__":
    # 1. 创建配置对象
    config = Config()
    # 2. 打印配置信息，验证是否正确加载
    print("根路径:", config.root_path)
    print("训练数据路径:", config.train_path)
    print("验证数据路径:", config.dev_path)
    print("测试数据路径:", config.test_path)
    print("停用词文件路径:", config.stopwords_path)
    print("类别文件路径:", config.class_path)
    print("类别映射:", config.id2class)
    # 3. 对训练、验证、测试数据集去重
    remove_duplicates(config.train_raw_path, config.train_path)
    remove_duplicates(config.dev_raw_path, config.dev_path)
    remove_duplicates(config.test_raw_path, config.test_path)
