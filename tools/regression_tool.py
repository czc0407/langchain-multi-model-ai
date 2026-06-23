"""
股价预测 LangChain Tool
"""
from langchain_core.tools import tool

from utils import parse_float_list, make_json_response
from models.rnn_regressor import RNNRegressor

_regressor = None


def _get_regressor():
    global _regressor
    if _regressor is None:
        _regressor = RNNRegressor()
    return _regressor


@tool
def predict_stock_price(historical_prices: str) -> str:
    """
    根据过去 20 步历史股价，预测下一步价格。

    参数:
        historical_prices: 逗号分隔的 20 个浮点数，代表过去 20 步的股价序列
    返回:
        JSON 字符串，包含预测价格、各 Agent 独立预测、Q-Learning 融合策略
    """
    try:
        seq = parse_float_list(historical_prices, expected_len=20)
        reg = _get_regressor()
        result = reg.predict(seq)
        return make_json_response(result)
    except Exception as e:
        return make_json_response({"error": str(e)})
