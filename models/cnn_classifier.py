"""
CNN 多 Agent 分类器封装
6 个 LeNetBinary 二分类 Agent + JudgeNet 二次保险 → 4 分类
"""
import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from config import (
    EXP1_CHECKPOINTS, PAIRS, PAIR_ORDER, CLASS_NAMES, CLASS_LABELS_ZH
)


# ─────────────────────────────────────────────
# 模型定义（复制自 exp1/model.py）
# ─────────────────────────────────────────────

class LeNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class JudgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 分类器封装（单例懒加载）
# ─────────────────────────────────────────────

class CNNClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._agents = {}
        for p1, p2 in PAIRS:
            key = f"{p1}{p2}"
            path = os.path.join(EXP1_CHECKPOINTS, f"agent_{key}.pth")
            model = LeNetBinary().to(self._device)
            model.load_state_dict(torch.load(path, map_location=self._device))
            model.eval()
            self._agents[key] = model

        judge_path = os.path.join(EXP1_CHECKPOINTS, "judge_net.pth")
        self._judge = JudgeNet().to(self._device)
        self._judge.load_state_dict(
            torch.load(judge_path, map_location=self._device)
        )
        self._judge.eval()
        self._loaded = True

    def predict(self, image_tensor):
        """
        参数:
            image_tensor: numpy array, shape (1, 1, 28, 28), float32, 值域 [0,1]
        返回:
            dict: {
                "class_id": int,       # 0-3
                "class_name": str,     # 英文名
                "class_name_zh": str,  # 中文名
                "vote_result": str,     # 投票结果 (A/B/C/D)
                "judge_result": int,    # JudgeNet 预测 (0-3)
                "vote_distribution": dict,  # 各类投票数
                "agent_probs": dict,    # 每个 agent 的 softmax 概率
            }
        """
        self._load()
        x = torch.from_numpy(image_tensor).to(self._device)

        with torch.no_grad():
            agent_probs = {}
            agent_preds = []  # 每个 agent 的 0/1 预测
            judge_feat = []

            for key in PAIR_ORDER:
                logits = self._agents[key](x)
                prob = softmax(logits, dim=1).squeeze(0)
                prob_list = prob.cpu().numpy().tolist()
                agent_probs[key] = {"class_0": float(prob_list[0]),
                                     "class_1": float(prob_list[1])}
                judge_feat.extend(prob_list)
                pred = int(logits.argmax(1).item())
                agent_preds.append(pred)

            # 多数投票
            votes = []
            for (p1, p2), pred in zip(PAIRS, agent_preds):
                votes.append(p1 if pred == 0 else p2)

            vote_count = {"A": 0, "B": 0, "C": 0, "D": 0}
            for v in votes:
                vote_count[v] += 1
            winner = max(vote_count, key=lambda x: vote_count[x])

            # Vote one-hot
            vote_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            vote_label = vote_map[winner]
            vote_onehot = [0, 0, 0, 0]
            vote_onehot[vote_label] = 1
            judge_feat.extend(vote_onehot)

            # JudgeNet
            feat_tensor = torch.tensor(
                judge_feat, dtype=torch.float32
            ).unsqueeze(0).to(self._device)
            judge_logits = self._judge(feat_tensor)
            judge_pred = int(judge_logits.argmax(1).item())

        return {
            "class_id": judge_pred,
            "class_name": CLASS_NAMES[judge_pred],
            "class_name_zh": CLASS_LABELS_ZH[judge_pred],
            "vote_result": winner,
            "judge_result": judge_pred,
            "vote_distribution": vote_count,
            "agent_probs": agent_probs,
        }
