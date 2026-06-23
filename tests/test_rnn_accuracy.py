"""
验证 RNN 模块准确率 >= 98%
首运行会自动训练并保存模型
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.rnn_regressor import RNNRegressor

print("=" * 60)
print("  RNN 准确率测试")
print("=" * 60)

print("\n[1/2] 加载 RNN 回归器（首次运行将训练模型）...")
regressor = RNNRegressor()

print("\n[2/2] 测试集评估...")
accuracy, metrics = regressor.evaluate()

print(f"\n{'=' * 60}")
print(f"  融合准确率: {accuracy*100:.2f}% ({metrics['correct']}/{metrics['total']})")
print(f"  单 Agent 准确率:")
names = ["LSTM", "GRU", "CNN+LSTM", "BiLSTM"]
for name, acc in zip(names, metrics["single_agent_accs"]):
    print(f"    {name:10s}: {acc*100:.2f}%")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAPE: {metrics['mape']:.2f}%")
print(f"  阈值: >= 98.00%")
if accuracy >= 0.98:
    print(f"  结果: PASS ✓")
else:
    print(f"  结果: FAIL ✗")
print(f"{'=' * 60}")

assert accuracy >= 0.98, (
    f"RNN 准确率 {accuracy*100:.2f}% 未达标（需 >= 98%）"
)
