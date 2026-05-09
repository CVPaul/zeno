# zeno

一个极简、本地优先的 agent 框架和命令行工具。

默认后端集中到 **vLLM family**：Mac M 系列用 `vllm-mlx`，Linux/WSL CUDA 用 `vllm`。Zeno 负责启动本地 OpenAI-compatible server，用户日常只操作 `zeno`。

## 安装

Mac M / Apple Silicon：

```bash
python -m pip install '.[mac]'
```

Linux / WSL + NVIDIA：

```bash
python -m pip install '.[cuda]'
```

只安装 Zeno 本体用于开发/测试：

```bash
python -m pip install .
```

## 快速开始

```bash
zeno
```

Zeno 会按平台选择后端：

- macOS arm64：`vllm-mlx`
- 其它平台：`vllm`

也可以手动指定：

```bash
zeno --backend vllm-mlx
zeno --backend vllm
```

## 任务命令

执行一次性任务：

```bash
zeno task create "实现一个简单的 numpy MLP 训练脚本"
```

指定模型：

```bash
zeno --backend vllm-mlx --model mlx-community/Qwen2.5-14B-Instruct-4bit task create "实现一个简单的 numpy MLP 训练脚本"
zeno --backend vllm --model Qwen/Qwen2.5-14B-Instruct task create "实现一个简单的 numpy MLP 训练脚本"
```

查看任务历史：

```bash
zeno task list
```

继续当前目录里的上一个任务：

```bash
zeno --continue
```

## 默认模型

Mac M / `vllm-mlx`：

```text
mlx-community/Qwen2.5-7B-Instruct-4bit
```

Linux/WSL / `vllm`：

```text
Qwen/Qwen2.5-7B-Instruct
```

## Python API

```python
from zeno import Agent, OpenAICompatibleChatModel

agent = Agent(
    model=OpenAICompatibleChatModel(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-7B-Instruct",
    ),
    system="你是一个简洁、可靠的助手。",
)

print(agent.run("用一句话解释什么是 agent。"))
```

## 工作区隔离

Zeno 的任务历史默认保存在当前启动目录：

```text
.zeno/sessions/
```

不同项目目录之间历史互不影响。

## 开发验证

```bash
python -m unittest discover -s tests
python -m py_compile examples/chat.py examples/tool_use.py
python main.py --help
python main.py task list
python main.py task --help
```

## 保留的可选 adapter

代码里仍保留 `OllamaChatModel`、`MLXChatModel` 和 `OpenAICompatibleChatModel`，方便实验；但 CLI 默认路线是 vLLM family。

## 非目标

Zeno 不内置 RAG、向量数据库、长期记忆、插件市场或复杂工作流编排。

## License

MIT
