"""
RNN 多 Agent 回归器封装
4 个 Keras Agent（LSTM/GRU/CNN+LSTM/BiLSTM）+ Q-Learning 融合
首次运行自动训练并保存，后续加载缓存
"""
import os
import pickle
import numpy as np
from math import factorial

from config import (
    EXP2_MODELS_DIR, EXP2_DATA_DIR,
    RNN_TIME_STEPS, RNN_SEQ_LEN, RNN_AHEAD, RNN_THRESHOLD,
    RNN_NOISE_STD, RNN_EPOCHS, RNN_BATCH_SIZE, RL_N_EPISODES,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class RNNRegressor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── 保存路径 ──
    def _paths(self):
        d = EXP2_MODELS_DIR
        return {
            "agent1": os.path.join(d, "agent1_lstm.keras"),
            "agent2": os.path.join(d, "agent2_gru.keras"),
            "agent3": os.path.join(d, "agent3_cnn_lstm.keras"),
            "agent4": os.path.join(d, "agent4_bilstm.keras"),
            "q_table":    os.path.join(d, "q_table.npy"),
            "act_wts":    os.path.join(d, "action_weights.npy"),
            "scaler_x":   os.path.join(d, "scaler_x.pkl"),
            "scaler_y":   os.path.join(d, "scaler_y.pkl"),
            "data":       os.path.join(EXP2_DATA_DIR, "stock_data.npz"),
        }

    def _all_saved(self):
        p = self._paths()
        return all(os.path.exists(v) for v in p.values())

    # ── 构建 4 个 Agent ──
    def _build_agents(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, GRU, Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional,
        )

        def a1():
            m = Sequential([
                LSTM(64, return_sequences=True, input_shape=(RNN_SEQ_LEN, 1)),
                Dropout(0.1),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1),
            ], name="Agent1_LSTM")
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            return m

        def a2():
            m = Sequential([
                GRU(64, return_sequences=True, input_shape=(RNN_SEQ_LEN, 1)),
                Dropout(0.1),
                GRU(32),
                Dense(16, activation="relu"),
                Dense(1),
            ], name="Agent2_GRU")
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            return m

        def a3():
            m = Sequential([
                Conv1D(32, 3, activation="relu", input_shape=(RNN_SEQ_LEN, 1)),
                MaxPooling1D(2),
                Conv1D(64, 3, activation="relu"),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1),
            ], name="Agent3_CNN_LSTM")
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            return m

        def a4():
            m = Sequential([
                Bidirectional(LSTM(64, return_sequences=True),
                              input_shape=(RNN_SEQ_LEN, 1)),
                Dropout(0.1),
                Bidirectional(LSTM(32)),
                Dense(16, activation="relu"),
                Dense(1),
            ], name="Agent4_BiLSTM")
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            return m

        return [a1(), a2(), a3(), a4()]

    # ── 训练 + 保存 ──
    def _train_and_save(self):
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler

        np.random.seed(42)
        tf.random.set_seed(42)

        print("[RNN] 首次运行，训练模型...")

        # ── 生成数据 ──
        t = np.arange(RNN_TIME_STEPS)
        trend = 0.003 * t
        s1 = 8.0 * np.sin(2 * np.pi * t / 120)
        s2 = 3.0 * np.sin(2 * np.pi * t / 40)
        s3 = 1.5 * np.sin(2 * np.pi * t / 25)
        noise = np.random.randn(RNN_TIME_STEPS) * RNN_NOISE_STD
        raw = trend + s1 + s2 + s3 + noise + 100

        X_all, y_all = [], []
        for i in range(RNN_SEQ_LEN, len(raw) - RNN_AHEAD):
            X_all.append(raw[i - RNN_SEQ_LEN : i])
            y_all.append(raw[i + RNN_AHEAD])
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_s = scaler_x.fit_transform(X_all)
        y_s = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        n = len(X_s)
        s1_idx = int(0.60 * n)
        s2_idx = int(0.80 * n)
        X_train, y_train = X_s[:s1_idx], y_s[:s1_idx]
        X_val,   y_val   = X_s[s1_idx:s2_idx], y_s[s1_idx:s2_idx]

        X_train3 = X_train.reshape(-1, RNN_SEQ_LEN, 1)
        X_val3   = X_val.reshape(-1, RNN_SEQ_LEN, 1)

        # 缓存数据
        np.savez(self._paths()["data"],
                 X_all=X_all, y_all=y_all,
                 train_end=s1_idx, val_end=s2_idx)

        # ── 训练 4 Agent ──
        agents = self._build_agents()
        names = ["LSTM", "GRU", "CNN+LSTM", "BiLSTM"]
        for i, (agent, name) in enumerate(zip(agents, names)):
            print(f"  [{i+1}/4] {name} 训练中...")
            agent.fit(X_train3, y_train,
                      epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE,
                      validation_data=(X_val3, y_val), verbose=0)
            print(f"  [{i+1}/4] {name} ✓")

        # 验证集预测
        preds_val = np.column_stack(
            [a.predict(X_val3, verbose=0).flatten() for a in agents]
        )

        # ── Q-Learning ──
        ACTION_WEIGHTS = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.55, 0.20, 0.15, 0.10],
            [0.20, 0.55, 0.15, 0.10],
            [0.15, 0.20, 0.55, 0.10],
            [0.10, 0.15, 0.20, 0.55],
            [0.15, 0.35, 0.15, 0.35],
            [0.10, 0.40, 0.40, 0.10],
            [0.30, 0.30, 0.20, 0.20],
        ])
        N_ACTIONS = ACTION_WEIGHTS.shape[0]

        def _rank_to_state(rank):
            s, items = 0, list(range(4))
            for pos, r in enumerate(rank):
                idx = items.index(r)
                s += idx * factorial(3 - pos)
                items.pop(idx)
            return s

        def get_state(wp, wt):
            mse = np.mean((wp - wt.reshape(-1, 1)) ** 2, axis=0)
            return _rank_to_state(tuple(np.argsort(np.argsort(mse))))

        def get_state_no_labels(wp):
            consensus = np.mean(wp, axis=1)
            dev = np.mean((wp - consensus.reshape(-1, 1)) ** 2, axis=0)
            return _rank_to_state(tuple(np.argsort(np.argsort(dev))))

        def compute_reward(wp, wt, a):
            fused = wp @ ACTION_WEIGHTS[a]
            fused_r = scaler_y.inverse_transform(fused.reshape(-1, 1)).flatten()
            true_r = scaler_y.inverse_transform(wt.reshape(-1, 1)).flatten()
            re = np.abs(fused_r - true_r) / (np.abs(true_r) + 1e-8)
            return float(np.mean(re <= RNN_THRESHOLD))

        Q = np.zeros((24, N_ACTIONS))
        ALPHA, GAMMA, EPSILON, DECAY = 0.3, 0.9, 0.5, 0.97
        WINDOW = 25
        val_len = len(y_val)

        print(f"  Q-Learning 训练中（{RL_N_EPISODES}轮）...")
        for ep in range(RL_N_EPISODES):
            i = 0
            while i + WINDOW <= val_len:
                wp, wt = preds_val[i:i+WINDOW], y_val[i:i+WINDOW]
                s = get_state(wp, wt)
                if np.random.rand() < EPSILON:
                    a = np.random.randint(N_ACTIONS)
                else:
                    a = int(np.argmax(Q[s]))
                r = compute_reward(wp, wt, a)
                ni = i + WINDOW
                bn = float(np.max(Q[get_state(
                    preds_val[ni:ni+WINDOW], y_val[ni:ni+WINDOW]
                )])) if ni + WINDOW <= val_len else 0.0
                Q[s, a] += ALPHA * (r + GAMMA * bn - Q[s, a])
                i += WINDOW
            EPSILON *= DECAY
        print("  Q-Learning ✓")

        # ── 保存 ──
        p = self._paths()
        for i, a in enumerate(agents):
            a.save(p[f"agent{i+1}"])
        np.save(p["q_table"], Q)
        np.save(p["act_wts"], ACTION_WEIGHTS)
        with open(p["scaler_x"], "wb") as f:
            pickle.dump(scaler_x, f)
        with open(p["scaler_y"], "wb") as f:
            pickle.dump(scaler_y, f)
        print("[RNN] 模型已保存\n")

    # ── 加载 ──
    def _load_all(self):
        import tensorflow as tf
        p = self._paths()
        self._agents = [
            tf.keras.models.load_model(p["agent1"]),
            tf.keras.models.load_model(p["agent2"]),
            tf.keras.models.load_model(p["agent3"]),
            tf.keras.models.load_model(p["agent4"]),
        ]
        self._q_table = np.load(p["q_table"])
        self._action_weights = np.load(p["act_wts"])
        with open(p["scaler_x"], "rb") as f:
            self._scaler_x = pickle.load(f)
        with open(p["scaler_y"], "rb") as f:
            self._scaler_y = pickle.load(f)
        self._names = ["LSTM", "GRU", "CNN+LSTM", "BiLSTM"]
        self._loaded = True

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self._all_saved():
            self._train_and_save()
        self._load_all()

    # ── 预测 ──
    def predict(self, sequence):
        """
        参数:
            sequence: list/array of 20 floats（历史价格序列）
        返回:
            dict: {
                "predicted_price": float,  # 反归一化后的预测值
                "agent_predictions": dict, # 各 agent 独立预测（反归一化）
                "action_index": int,       # Q-Learning 选择的策略索引
                "action_weights": list,    # 策略权重
                "strategy_desc": str,      # 策略描述
            }
        """
        self._ensure_loaded()
        seq = np.array(sequence, dtype=np.float32).reshape(1, -1)
        scaled = self._scaler_x.transform(seq).reshape(1, RNN_SEQ_LEN, 1)

        # 各 agent 独立预测（scaled）
        raw_preds = np.array(
            [a.predict(scaled, verbose=0).flatten()[0] for a in self._agents]
        )

        # 状态估计：用 4 agent 间的 consensus 偏离度
        wp = raw_preds.reshape(1, -1)
        consensus = float(np.mean(wp))
        dev = np.mean((wp - consensus) ** 2, axis=0)
        rank = tuple(np.argsort(np.argsort(dev)))

        # Lehmer code → state index
        def _rank_to_state(rank):
            s, items = 0, list(range(4))
            for pos, r in enumerate(rank):
                idx = items.index(r)
                s += idx * factorial(3 - pos)
                items.pop(idx)
            return s

        state = _rank_to_state(rank)
        action = int(np.argmax(self._q_table[state]))
        weights = self._action_weights[action]

        # 加权融合
        fused_scaled = float(np.dot(raw_preds, weights))
        predicted = float(
            self._scaler_y.inverse_transform([[fused_scaled]]).flatten()[0]
        )

        # 各 agent 独立预测反归一化
        agent_preds_unscaled = {
            name: float(
                self._scaler_y.inverse_transform([[p]]).flatten()[0]
            )
            for name, p in zip(self._names, raw_preds)
        }

        strategy_descs = [
            "均等加权",
            "偏重 LSTM",
            "偏重 GRU",
            "偏重 CNN+LSTM",
            "偏重 BiLSTM",
            "GRU+BiLSTM 均衡",
            "GRU+CNN 均衡",
            "略偏 LSTM+GRU",
        ]
        desc = strategy_descs[action] if action < len(strategy_descs) else f"策略{action}"

        return {
            "predicted_price": round(predicted, 4),
            "agent_predictions": {
                k: round(v, 4) for k, v in agent_preds_unscaled.items()
            },
            "action_index": action,
            "action_weights": [round(w, 4) for w in weights.tolist()],
            "strategy_desc": desc,
        }

    # ── 评估 ──
    def evaluate(self):
        """在测试集上评估准确率，返回 (accuracy, metrics_dict)"""
        self._ensure_loaded()

        # 加载测试数据
        data = np.load(self._paths()["data"])
        X_all, y_all = data["X_all"], data["y_all"]
        val_end = int(data["val_end"])

        X_test = X_all[val_end:]
        y_test = y_all[val_end:]
        X_test_s = self._scaler_x.transform(X_test).reshape(-1, RNN_SEQ_LEN, 1)

        # 各 agent 独立预测
        preds_test = np.column_stack([
            a.predict(X_test_s, verbose=0).flatten() for a in self._agents
        ])

        # Q-Learning 融合（consensus-based state）
        test_len = len(y_test)
        WINDOW = 25
        y_pred_final = np.zeros(test_len)
        i = 0
        last_action = int(np.argmax(self._q_table.sum(axis=0)))

        def _rank_to_state(rank):
            s, items = 0, list(range(4))
            for pos, r in enumerate(rank):
                idx = items.index(r)
                s += idx * factorial(3 - pos)
                items.pop(idx)
            return s

        while i < test_len:
            end = min(i + WINDOW, test_len)
            wp = preds_test[i:end]
            if len(wp) >= 5:
                consensus = np.mean(wp, axis=1)
                dev = np.mean((wp - consensus.reshape(-1, 1)) ** 2, axis=0)
                rank = tuple(np.argsort(np.argsort(dev)))
                s = _rank_to_state(rank)
                a = int(np.argmax(self._q_table[s]))
                last_action = a
            else:
                a = last_action
            y_pred_final[i:end] = wp @ self._action_weights[a]
            i += WINDOW

        y_pred_orig = self._scaler_y.inverse_transform(
            y_pred_final.reshape(-1, 1)
        ).flatten()
        y_test_orig = self._scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        rel_errors = np.abs(y_pred_orig - y_test_orig) / (
            np.abs(y_test_orig) + 1e-8
        )
        correct = int(np.sum(rel_errors <= RNN_THRESHOLD))
        total = test_len
        accuracy = correct / total
        mae = float(np.mean(np.abs(y_pred_orig - y_test_orig)))
        rmse = float(np.sqrt(np.mean((y_pred_orig - y_test_orig) ** 2)))
        mape = float(np.mean(rel_errors) * 100)

        # 单 agent 准确率
        single_accs = []
        for i in range(4):
            p_o = self._scaler_y.inverse_transform(
                preds_test[:, i].reshape(-1, 1)
            ).flatten()
            re = np.abs(p_o - y_test_orig) / (np.abs(y_test_orig) + 1e-8)
            single_accs.append(float(np.mean(re <= RNN_THRESHOLD)))

        return accuracy, {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "single_agent_accs": single_accs,
        }
