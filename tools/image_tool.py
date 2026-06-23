"""
图像分类 LangChain Tool
"""
import os
from langchain_core.tools import tool

from utils import img_to_tensor, make_json_response
from models.cnn_classifier import CNNClassifier

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = CNNClassifier()
    return _classifier


@tool
def classify_fashion_image(image_path: str) -> str:
    """
    对 Fashion-MNIST 风格的服装图片进行分类。
    支持 T-shirt/top (T恤)、Trouser (裤子)、Sneaker (运动鞋)、Bag (包) 四类。

    参数:
        image_path: 本地图片文件路径，图片会被自动转换为 28×28 灰度图
    返回:
        JSON 字符串，包含分类结果、投票分布和各 Agent 概率
    """
    if not os.path.exists(image_path):
        return make_json_response({"error": f"文件不存在: {image_path}"})

    try:
        tensor = img_to_tensor(image_path)
        clf = _get_classifier()
        result = clf.predict(tensor)
        return make_json_response(result)
    except Exception as e:
        return make_json_response({"error": str(e)})
