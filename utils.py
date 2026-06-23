"""
公共工具函数
"""
import json
import numpy as np
from PIL import Image


def img_to_tensor(image_path, target_size=28):
    """将任意图片转为 28×28 灰度归一化 tensor (1, 1, 28, 28)
    返回 numpy float32 数组，shape=(1, 1, 28, 28)，值域 [0, 1]
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, target_size, target_size)


def parse_float_list(s, expected_len=20):
    """将逗号/空格分隔的字符串解析为 float 列表"""
    s = s.replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != expected_len:
        raise ValueError(
            f"需要恰好 {expected_len} 个数，收到 {len(vals)} 个"
        )
    return vals


def make_json_response(data: dict) -> str:
    """将 dict 转为格式化 JSON 字符串"""
    return json.dumps(data, ensure_ascii=False, indent=2)
