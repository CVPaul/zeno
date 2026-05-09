from zeno import Agent, ensure_default_local_model


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = Agent(
    model=ensure_default_local_model(),
    system="你是一个简洁的助手。需要计算时可以调用可用工具。",
    tools={"add": add},
)


if __name__ == "__main__":
    print(agent.run("请计算 19 + 23，并给出一句话答案。"))
