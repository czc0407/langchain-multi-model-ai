"""
LangChain Agent 配置：LLM + Tools + 对话管理
"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    LLM_PROVIDER, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY,
    LLM_MODEL_NAME, LLM_TEMPERATURE, MEMORY_WINDOW_K,
)
from tools.image_tool import classify_fashion_image
from tools.regression_tool import predict_stock_price

TOOLS = [classify_fashion_image, predict_stock_price]


def _get_llm():
    if LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=LLM_TEMPERATURE,
        )
    else:
        from langchain_community.chat_models.tongyi import ChatTongyi
        return ChatTongyi(
            model=LLM_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            temperature=LLM_TEMPERATURE,
        )


SYSTEM_PROMPT = """你是一个严格的多模型 AI 助手，只能通过调用工具完成任务。你必须遵守以下规则：

1. **只能使用提供的 2 个工具**，绝对不自行猜测或编造结果。

2. **路由规则：**
   - 用户询问图片分类、服装识别、"这是什么衣服"、"classify"、"识别"等图像相关请求 → 调用 classify_fashion_image
   - 用户询问价格预测、数值预测、"predict price"、"forecast"、"预测股价"等时序预测请求 → 调用 predict_stock_price
   - 用户没有指定本地图片路径时，引导用户提供图片路径
   - 用户需要预测但没有提供 20 个历史价格时，引导用户提供逗号分隔的 20 个数字

3. **无关请求处理：**
   - 与图片分类或价格预测无关的请求 → 礼貌拒绝，说明你只能处理服装图片分类和股价预测

4. **结果展示规则（极其重要）：**
   - 工具返回的数值结果（分类结果、预测价格、概率值等）**绝对不能修改**
   - 你可以用自然语言解释结果，但必须原样引用数值
   - 如有错误信息，直接向用户展示

5. **语言匹配：** 用与用户相同的语言回复"""


class AgentSession:
    """封装 Agent 和多轮对话状态"""

    def __init__(self):
        llm = _get_llm()
        self._agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
        self._history: list = []  # HumanMessage / AIMessage

    def chat(self, user_input: str) -> str:
        """发送用户消息，返回 Agent 回复文本"""
        messages = list(self._history)
        messages.append(HumanMessage(content=user_input))

        result = self._agent.invoke({"messages": messages})

        # 提取所有 AI 回复
        responses = [
            msg.content for msg in result.get("messages", [])
            if isinstance(msg, AIMessage) and msg.content
        ]

        # 取最后一条 AI 回复
        response = responses[-1] if responses else "(无回复)"

        # 更新对话历史（截断到最近 k 轮，每轮 = human + ai）
        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=response))
        max_msgs = MEMORY_WINDOW_K * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

        return response


def create_agent_session():
    """创建并返回 AgentSession"""
    return AgentSession()
