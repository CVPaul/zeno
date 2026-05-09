from zeno import Agent, ensure_default_local_model


agent = Agent(
    model=ensure_default_local_model(),
    system="你是一个简洁、可靠的中文助手。",
)


if __name__ == "__main__":
    print("输入 exit / quit / 空行退出。")
    while True:
        try:
            prompt = input("你> ").strip()
        except EOFError:
            break
        if prompt in {"", "exit", "quit"}:
            break
        print(f"agent> {agent.run(prompt)}")
