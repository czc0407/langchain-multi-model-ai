"""
入口：交互式 CLI demo
用法:
    python main.py              # 交互模式
    python main.py --demo       # 运行预设测试用例
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import create_agent_session
from config import LLM_PROVIDER

DEMO_CASES = [
    # ── 图片分类测试 ──
    "请帮我看一下这张图片是什么衣服？图片路径是 /tmp/test_fashion.png",

    # ── 股价预测测试 ──
    "预测下一步股价：100.5,101.2,102.1,101.8,103.4,104.2,105.1,104.8,106.3,"
    "105.9,107.2,108.1,107.6,109.3,110.1,109.8,111.2,112.4,113.1,112.9",

    # ── 无关请求 ──
    "今天天气怎么样？",

    # ── 英文股价 ──
    "I have a stock sequence, can you predict: "
    "50.2,50.8,51.3,51.9,52.4,53.1,53.8,54.2,54.9,55.6,"
    "56.1,56.8,57.3,58.0,58.5,59.2,59.8,60.3,60.9,61.5",
]


def run_interactive(session):
    print("=" * 60)
    print("  LangChain 多模型 AI 系统 - 交互式 Demo")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print("  支持：服装图片分类 (classify_fashion_image)")
    print("        股价预测     (predict_stock_price)")
    print("  输入 'quit' 或 Ctrl+C 退出")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            print("Agent > ", end="", flush=True)
            response = session.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] {e}\n")


def run_demo(session):
    print("=" * 60)
    print("  LangChain 多模型 AI 系统 - Demo 模式")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print("=" * 60)
    print()

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"--- 测试用例 {i}/{len(DEMO_CASES)} ---")
        print(f"User > {case}")
        print(f"Agent > ", end="", flush=True)

        try:
            response = session.chat(case)
            print(response)
        except Exception as e:
            print(f"[错误] {e}")
        print()


def main():
    print("[系统] 初始化 Agent...")
    session = create_agent_session()
    print("[系统] 初始化完成\n")

    if "--demo" in sys.argv:
        run_demo(session)
    else:
        run_interactive(session)


if __name__ == "__main__":
    main()
