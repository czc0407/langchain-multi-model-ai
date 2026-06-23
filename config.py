"""
全局配置：路径、LLM、超参数
"""
import os

# ── 项目根目录 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 数据 / 模型路径 ──
EXP1_CHECKPOINTS = os.path.join(BASE_DIR, "data", "exp1", "checkpoints")
EXP2_MODELS_DIR  = os.path.join(BASE_DIR, "data", "exp2", "models")
EXP2_DATA_DIR    = os.path.join(BASE_DIR, "data", "exp2", "data")

# ── CNN 配置 ──
CLASS_MAP = {0: 0, 1: 1, 7: 2, 8: 3}       # FashionMNIST → 内部索引
CLASS_NAMES = {0: "T-shirt/top", 1: "Trouser", 2: "Sneaker", 3: "Bag"}
CLASS_LABELS_ZH = {0: "T恤", 1: "裤子", 2: "运动鞋", 3: "包"}
PAIRS = [("A","B"), ("A","C"), ("A","D"), ("B","C"), ("B","D"), ("C","D")]
PAIR_ORDER = ["AB", "AC", "AD", "BC", "BD", "CD"]

# ── RNN 配置 ──
RNN_TIME_STEPS = 8000
RNN_SEQ_LEN    = 20
RNN_AHEAD      = 1
RNN_THRESHOLD  = 0.02
RNN_NOISE_STD  = 0.4
RNN_EPOCHS     = 30
RNN_BATCH_SIZE = 64
RL_N_EPISODES  = 150

# ── LLM 配置 ──
LLM_PROVIDER       = os.environ.get("LLM_PROVIDER", "dashscope")
DASHSCOPE_API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL_NAME     = os.environ.get("LLM_MODEL_NAME", "qwen-plus")
LLM_TEMPERATURE    = 0.1

# ── Agent 配置 ──
MAX_ITERATIONS     = 3
MEMORY_WINDOW_K    = 5
