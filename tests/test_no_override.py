"""
验证 LLM 不篡改模型输出
确保 Agent 的 System Prompt 包含"不修改数值"规则
并通过直接调用 Tools 验证其输出格式和数值一致性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
import numpy as np
from PIL import Image

from agent import SYSTEM_PROMPT, TOOLS
from utils import img_to_tensor
from models.cnn_classifier import CNNClassifier
from models.rnn_regressor import RNNRegressor

print("=" * 60)
print("  模型输出防篡改验证")
print("=" * 60)

# ── 1. 验证 System Prompt 包含防篡改规则 ──
print("\n[1/4] 检查 System Prompt 防篡改规则...")
anti_override_rules = [
    "不能修改",
    "绝对不修改",
    "不能自行猜测",
    "原样",
]
for rule in anti_override_rules:
    assert any(
        rule in SYSTEM_PROMPT or rule.lower() in SYSTEM_PROMPT.lower()
        for rule in anti_override_rules
    ), "防篡改规则至少需要一条"
print(f"  ✓ 防篡改规则存在")

# ── 2. 验证图像分类 Tool 输出格式 ──
print("\n[2/4] 验证 classify_fashion_image 输出格式...")

# 创建一张测试用的 28×28 随机灰度图
tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
tmp_path = tmp_img.name
rand_img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
Image.fromarray(rand_img, mode="L").save(tmp_path)

try:
    result_str = classify_fashion_image.invoke({"image_path": tmp_path})
    result = json.loads(result_str)
    required_keys = ["class_id", "class_name", "class_name_zh",
                     "vote_result", "judge_result",
                     "vote_distribution", "agent_probs"]
    for key in required_keys:
        assert key in result, f"缺少字段: {key}"
    # 验证 class_id 是 0-3 的整数
    assert isinstance(result["class_id"], int) and 0 <= result["class_id"] <= 3
    # 验证 vote_distribution 总和为 6
    assert sum(result["vote_distribution"].values()) == 6
    # 验证 agent_probs 有 6 对 agent
    assert len(result["agent_probs"]) == 6
    print(f"  ✓ 图像分类输出格式正确")
finally:
    os.unlink(tmp_path)

# ── 3. 验证股价预测 Tool 输出格式 ──
print("\n[3/4] 验证 predict_stock_price 输出格式...")

# 生成 20 个随机价格
test_prices = [round(100 + np.sin(i * 0.3) * 10 + np.random.randn() * 2, 2)
               for i in range(20)]
prices_str = ", ".join(str(p) for p in test_prices)

result_str = predict_stock_price.invoke({"historical_prices": prices_str})
result = json.loads(result_str)
required_keys = ["predicted_price", "agent_predictions",
                 "action_index", "action_weights", "strategy_desc"]
for key in required_keys:
    assert key in result, f"缺少字段: {key}"
assert isinstance(result["predicted_price"], (int, float))
assert len(result["agent_predictions"]) == 4
assert isinstance(result["action_index"], int) and 0 <= result["action_index"] <= 7
assert len(result["action_weights"]) == 4
print(f"  ✓ 股价预测输出格式正确")

# ── 4. 验证同一输入产生一致输出（确定性） ──
print("\n[4/4] 验证预测结果确定性...")

# 多次调用应返回相同结果
first_result = predict_stock_price.invoke({"historical_prices": prices_str})
second_result = predict_stock_price.invoke({"historical_prices": prices_str})
assert first_result == second_result, "同一输入产生不一致输出"
print(f"  ✓ 同一输入输出一致（确定性）")

print(f"\n{'=' * 60}")
print(f"  所有防篡改检查通过 ✓")
print(f"{'=' * 60}")
