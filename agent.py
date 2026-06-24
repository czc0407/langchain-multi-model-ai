"""
LangChain Agent 配置：LLM + Tools + Memory
"""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import (
    LLM_PROVIDER, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY,
    LLM_MODEL_NAME, LLM_TEMPERATURE, MAX_ITERATIONS, MEMORY_WINDOW_K,
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
        # dashscope (通义千问)
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

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def create_agent():
    """创建并返回 AgentExecutor"""
    llm = _get_llm()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=MEMORY_WINDOW_K,
    )
    agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        max_iterations=MAX_ITERATIONS,
        verbose=False,
        handle_parsing_errors=True,
    )
