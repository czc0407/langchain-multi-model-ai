# LangChain 多模型 AI 系统

基于 LangChain 框架，集成 CNN 图像分类和 RNN 时序回归模型，使用 LLM 作为路由和解释层的多模型智能系统。

## 系统架构

```
用户自然语言输入
    │
    ▼
┌─────────────────────────────────────────┐
│  LangChain Agent (LLM: 通义千问/DeepSeek) │
│  - 意图理解 & 路由决策                    │
│  - 结果解释 & 格式化输出                  │
│  - 不篡改模型预测结果                     │
└────────┬──────────────┬─────────────────┘
         │              │
    ┌────▼─────┐  ┌─────▼──────┐
    │ Image Tool │  │ Stock Tool  │
    │ (CNN分类)  │  │ (RNN回归)   │
    └────┬─────┘  └─────┬──────┘
         │              │
    ┌────▼──────────┐  ┌▼──────────────────┐
    │ 6×LeNet 二分类 │  │ 4×RNN Agent      │
    │ + 投票 + Judge │  │ (LSTM/GRU/BiLSTM) │
    │   → 4分类输出  │  │ + Q-Learning 融合  │
    └───────────────┘  └───────────────────┘
```

## 功能

- **服装图片分类**：6 个 LeNet 二分类 Agent + 多数投票 + JudgeNet 二次保险，支持 T恤/裤子/运动鞋/包四分类
- **股价趋势预测**：4 个 RNN Agent（LSTM/GRU/CNN+LSTM/BiLSTM）+ Q-Learning 强化学习融合
- **自然语言交互**：用户用中文或英文描述需求，LLM 自动路由到对应工具
- **严格输出保护**：LLM 不修改模型预测数值，只做解释

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- TensorFlow 2.12+

## 安装

```bash
pip install -r requirements.txt
```

## LLM API 配置

```bash
# 方式一：通义千问
export DASHSCOPE_API_KEY="your-key"
export LLM_PROVIDER="dashscope"

# 方式二：DeepSeek
export DEEPSEEK_API_KEY="your-key"
export LLM_PROVIDER="deepseek"
```

## 使用方法

```bash
# 交互模式
python main.py

# Demo 模式（预设测试用例自动运行）
python main.py --demo
```

## 运行测试

```bash
# 验证 CNN 准确率（需 >= 98%）
python tests/test_cnn_accuracy.py

# 验证 RNN 准确率（需 >= 98%，首次运行会自动训练并保存模型）
python tests/test_rnn_accuracy.py

# 验证路由准确性（需 100%）
python tests/test_routing.py

# 验证 LLM 不篡改模型输出
python tests/test_no_override.py
```

## 项目结构

```
langchain_system/
├── main.py                     # 入口：CLI 交互式 demo
├── agent.py                    # LangChain Agent 配置
├── config.py                   # 全局配置
├── requirements.txt            # 依赖
├── utils.py                    # 工具函数
├── tools/
│   ├── image_tool.py           # 图像分类 LangChain Tool
│   └── regression_tool.py      # 股价预测 LangChain Tool
├── models/
│   ├── cnn_classifier.py       # CNN 分类器封装
│   └── rnn_regressor.py        # RNN 回归器封装
├── tests/
│   ├── test_cnn_accuracy.py
│   ├── test_rnn_accuracy.py
│   ├── test_routing.py
│   └── test_no_override.py
└── data/
    ├── exp1/                   # 软链接到实验一 checkpoints
    └── exp2/                   # RNN 训练权重（首次运行自动生成）
```

## 技术栈

| 组件 | 实现 |
|------|------|
| Agent 框架 | LangChain `create_agent` |
| LLM | 通义千问 (DashScope) / DeepSeek |
| CNN 模型 | PyTorch LeNet (6 agent + JudgeNet) |
| RNN 模型 | TensorFlow/Keras (LSTM/GRU/CNN+LSTM/BiLSTM) |
| 融合策略 | Q-Learning 强化学习 (24 状态 × 8 动作) |
| 数据集 | FashionMNIST (CNN) / 自模拟股票数据 (RNN) |
