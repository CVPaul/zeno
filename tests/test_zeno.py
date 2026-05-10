from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from zeno import Agent, ChatResponse, ConfigStore, DEFAULT_VLLM_MODEL, MLXChatModel, Message, OllamaChatModel, OllamaManager, OpenAICompatibleChatModel, SessionStore, ToolCall, VllmFamilyManager, clean_model_output, default_backend, default_local_model, default_model_name, default_tools, parse_inline_tool_calls, split_model_output, strip_inline_tool_calls, tool_schema
from zeno.cli import compact_messages, default_agent, is_exit_prompt, main as cli_main, print_thinking, typewriter_print
from zeno.models import _parse_tool_calls, MLXChatModel as ConcreteMLXChatModel
from zeno.sessions import default_session_dir


class FakeModel:
    def __init__(self, replies: list[ChatResponse]) -> None:
        self.replies = replies
        self.messages: list[list[Message]] = []
        self.tools: list[list[dict[str, object]] | None] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        self.messages.append([message.copy() for message in messages])
        self.tools.append(tools)
        return self.replies.pop(0)


class FakeStreamingModel:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.messages: list[list[Message]] = []
        self.tools: list[list[dict[str, object]] | None] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        raise AssertionError("streaming path should not call chat")

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        self.messages.append([message.copy() for message in messages])
        self.tools.append(tools)
        yield from self.chunks


class AgentTests(unittest.TestCase):
    def test_returns_plain_model_reply(self) -> None:
        agent = Agent(model=FakeModel([ChatResponse("hello", [])]), system="test")

        self.assertEqual(agent.run("hi"), "hello")

    def test_strips_gemma_thought_channel(self) -> None:
        raw = '<|channel>thought\nThe user greeted us.\n<channel|>Hello!'
        agent = Agent(model=FakeModel([ChatResponse(raw, [])]), system="test")

        self.assertEqual(agent.run("hi"), "Hello!")

    def test_splits_gemma_thinking_from_answer(self) -> None:
        result = split_model_output('<|channel>thought\nThe user greeted us.\n<channel|>Hello!')

        self.assertEqual(result.thinking, "The user greeted us.")
        self.assertEqual(result.answer, "Hello!")

    def test_strips_xml_think_block(self) -> None:
        self.assertEqual(clean_model_output("<think>hidden</think>Visible"), "Visible")

    def test_executes_native_tool_call_then_returns_final_reply(self) -> None:
        model = FakeModel(
            [
                ChatResponse("", [ToolCall(name="add", arguments={"a": 2, "b": 3})]),
                ChatResponse("2 + 3 = 5", []),
            ]
        )
        agent = Agent(
            model=model,
            system="test",
            tools={"add": lambda a, b: a + b},
        )

        self.assertEqual(agent.run("sum"), "2 + 3 = 5")
        self.assertEqual(model.messages[1][-1]["role"], "tool")
        self.assertEqual(model.messages[1][-1]["tool_name"], "add")
        self.assertEqual(model.messages[1][-1]["content"], "5")
        self.assertIsNotNone(model.tools[0])

    def test_unknown_tool_raises(self) -> None:
        agent = Agent(
            model=FakeModel([ChatResponse("", [ToolCall(name="missing", arguments={})])]),
            tools={},
        )

        with self.assertRaisesRegex(RuntimeError, "Unknown tool"):
            agent.run("call")

    def test_default_write_file_tool_creates_workspace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = default_tools(Path(tmpdir))
            result = tools["write_file"]("scripts/train_mlp.py", "print('train')\n")

            created = Path(tmpdir) / "scripts" / "train_mlp.py"
            self.assertEqual(created.read_text(encoding="utf-8"), "print('train')\n")
            self.assertEqual(result["path"], "scripts/train_mlp.py")

    def test_default_write_file_tool_blocks_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = default_tools(Path(tmpdir))

            with self.assertRaisesRegex(RuntimeError, "inside the workspace"):
                tools["write_file"]("../outside.py", "bad")

    def test_parses_gemma_inline_tool_call(self) -> None:
        raw = 'I will write it.\n<|tool_call>call:write_file{content:<|"|>print("hi")\n<|"|>,path:<|"|>mlp_training.py<|"|>}<tool_call|>'

        calls = parse_inline_tool_calls(raw)

        self.assertEqual(calls, [ToolCall(name="write_file", arguments={"content": 'print("hi")\n', "path": "mlp_training.py"})])
        self.assertEqual(strip_inline_tool_calls(raw), "I will write it.")

    def test_executes_gemma_inline_write_file_tool(self) -> None:
        raw = 'I will write it.\n<|tool_call>call:write_file{content:<|"|>print("mlp")\n<|"|>,path:<|"|>mlp_training.py<|"|>}<tool_call|>'
        model = FakeModel([ChatResponse(raw, [])])

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(model=model, tools=default_tools(Path(tmpdir)))
            answer = agent.run("implement an mlp trainer")
            created = Path(tmpdir) / "mlp_training.py"

            self.assertEqual(created.read_text(encoding="utf-8"), 'print("mlp")\n')

        self.assertIn("I will write it.", answer)
        self.assertIn("tool write_file completed", answer)
        self.assertIn("mlp_training.py", answer)

    def test_retries_file_request_when_model_only_describes_code(self) -> None:
        model = FakeModel(
            [
                ChatResponse("You can save this as train_mlp.py.", []),
                ChatResponse('<|tool_call>call:write_file{path:<|"|>train_mlp.py<|"|>,content:<|"|>print("mlp")\n<|"|>}<tool_call|>', []),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(model=model, tools=default_tools(Path(tmpdir)))
            answer = agent.run("实现一个mlp训练脚本")

            self.assertEqual((Path(tmpdir) / "train_mlp.py").read_text(encoding="utf-8"), 'print("mlp")\n')

        self.assertIn("tool write_file completed", answer)
        self.assertEqual(len(model.messages), 2)
        self.assertIn("You did not call write_file", model.messages[1][-1]["content"])

    def test_streams_chunks_before_returning_result(self) -> None:
        seen: list[str] = []
        agent = Agent(model=FakeStreamingModel(["hel", "lo"]), system="test")

        result = agent.stream_messages_with_result([{"role": "user", "content": "hi"}], on_chunk=seen.append)

        self.assertIsNotNone(result)
        self.assertEqual(seen, ["hel", "lo"])
        self.assertEqual(result.answer, "hello")
        self.assertTrue(result.streamed)

    def test_streaming_hides_inline_tool_call_and_executes_it(self) -> None:
        chunks = [
            "Writing\n<|tool_",
            "call>call:write_file{path:<|\"|>mlp.py<|\"|>,content:<|\"|>print(1)\n<|\"|>}<tool_",
            "call|>",
        ]
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(model=FakeStreamingModel(chunks), tools=default_tools(Path(tmpdir)))
            result = agent.stream_messages_with_result([{"role": "user", "content": "write"}], on_chunk=seen.append)
            created = Path(tmpdir) / "mlp.py"

            self.assertEqual(created.read_text(encoding="utf-8"), "print(1)\n")

        self.assertIsNotNone(result)
        self.assertEqual("".join(seen), "Writing\ntool write_file completed: {\"path\": \"mlp.py\", \"bytes\": 9}")
        self.assertEqual(result.answer, "Writing\ntool write_file completed: {\"path\": \"mlp.py\", \"bytes\": 9}")

    def test_streaming_inline_tool_call_creates_file_without_duplicate_answer(self) -> None:
        chunks = [
            "好的，我将创建文件。\n<|tool_call>",
            "call:write_file{path:<|\"|>mlp_training.py<|\"|>,content:<|\"|>print('mlp')\n<|\"|>}",
            "<tool_call|>",
        ]
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(model=FakeStreamingModel(chunks), tools=default_tools(Path(tmpdir)))

            result = agent.stream_messages_with_result([{"role": "user", "content": "实现一个mlp训练脚本"}], on_chunk=seen.append)

            self.assertEqual((Path(tmpdir) / "mlp_training.py").read_text(encoding="utf-8"), "print('mlp')\n")

        self.assertIsNotNone(result)
        shown = "".join(seen)
        self.assertEqual(shown.count("好的，我将创建文件。"), 1)
        self.assertIn("tool write_file completed", shown)
        self.assertEqual(result.answer.count("好的，我将创建文件。"), 1)

    def test_streaming_retries_file_request_when_model_only_describes_code(self) -> None:
        class RecoveringStreamingModel(FakeStreamingModel):
            def __init__(self) -> None:
                super().__init__(["你可以将以下代码保存为 train_mlp.py。"])

            def chat(self, messages: list[Message], tools: list[dict[str, object]] | None = None) -> ChatResponse:
                self.messages.append([message.copy() for message in messages])
                self.tools.append(tools)
                return ChatResponse('<|tool_call>call:write_file{path:<|"|>train_mlp.py<|"|>,content:<|"|>print("mlp")\n<|"|>}<tool_call|>', [])

        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            model = RecoveringStreamingModel()
            agent = Agent(model=model, tools=default_tools(Path(tmpdir)))

            result = agent.stream_messages_with_result([{"role": "user", "content": "实现一个mlp训练脚本"}], on_chunk=seen.append)

            self.assertEqual((Path(tmpdir) / "train_mlp.py").read_text(encoding="utf-8"), 'print("mlp")\n')

        self.assertIsNotNone(result)
        self.assertIn("tool write_file completed", result.answer)
        self.assertIn("tool write_file completed", "".join(seen))
        self.assertIn("You did not call write_file", model.messages[1][-1]["content"])

    def test_streaming_shows_thinking_channel(self) -> None:
        chunks = ["<|channel>tho", "ught\nPlan", " step\n<channel|>", "Final"]
        seen: list[str] = []
        agent = Agent(model=FakeStreamingModel(chunks), system="test")

        result = agent.stream_messages_with_result([{"role": "user", "content": "hi"}], on_chunk=seen.append)

        self.assertIsNotNone(result)
        self.assertEqual(result.thinking, "Plan step")
        self.assertEqual(result.answer, "Final")
        self.assertIn("thinking:\n  Plan step\n", "".join(seen))
        self.assertTrue("".join(seen).endswith("Final"))

    def test_streaming_thinking_is_emitted_before_final_answer(self) -> None:
        chunks = ["<|channel>thought\nPlan", " step", "\n<channel|>Final"]
        seen: list[str] = []
        agent = Agent(model=FakeStreamingModel(chunks), system="test")

        result = agent.stream_messages_with_result([{"role": "user", "content": "hi"}], on_chunk=seen.append)

        self.assertIsNotNone(result)
        shown_before_final = "".join(seen[:-1])
        self.assertIn("thinking:\n  Plan step\n", "".join(seen))
        self.assertIn("Plan", shown_before_final)
        self.assertTrue(seen[-1].endswith("Final"))
        self.assertEqual(result.thinking, "Plan step")

    def test_tool_schema_uses_function_signature(self) -> None:
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        schema = tool_schema("add", add)

        function = schema["function"]
        self.assertIsInstance(function, dict)
        parameters = function["parameters"]
        self.assertIsInstance(parameters, dict)
        self.assertEqual(parameters["required"], ["a", "b"])


class ModelParsingTests(unittest.TestCase):
    def test_parse_openai_style_string_arguments(self) -> None:
        calls = _parse_tool_calls(
            [
                {
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ]
        )

        self.assertEqual(calls, [ToolCall(name="add", arguments={"a": 1, "b": 2})])

    def test_malformed_tool_arguments_raise(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "arguments are invalid JSON"):
            _parse_tool_calls([{"function": {"name": "add", "arguments": "{"}}])


class MLXChatModelTests(unittest.TestCase):
    def test_chat_uses_loaded_runtime_text(self) -> None:
        model = ConcreteMLXChatModel(model="test-model")
        model._runtime = FakeMLXRuntime("hello from mlx")

        response = model.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(response, ChatResponse("hello from mlx", []))

    def test_tools_fail_clearly_on_mlx_backend(self) -> None:
        model = ConcreteMLXChatModel(model="test-model")

        with self.assertRaisesRegex(RuntimeError, "does not support native tool calls"):
            model.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])


class FakeMLXRuntime:
    def __init__(self, text: str) -> None:
        self.text = text

    def chat(self, messages: list[Message], max_tokens: int | None = None) -> object:
        return type("Response", (), {"text": self.text})()


class CliTests(unittest.TestCase):
    def test_public_import_compatibility(self) -> None:
        self.assertEqual(Agent.__name__, "Agent")
        self.assertEqual(MLXChatModel.__name__, "MLXChatModel")
        self.assertEqual(OllamaChatModel.__name__, "OllamaChatModel")
        self.assertEqual(OpenAICompatibleChatModel.__name__, "OpenAICompatibleChatModel")
        self.assertIsInstance(default_local_model(), OpenAICompatibleChatModel)

    def test_default_model_is_vllm_family_model(self) -> None:
        self.assertEqual(default_backend(), "vllm")
        self.assertEqual(default_model_name(), DEFAULT_VLLM_MODEL)
        self.assertEqual(default_local_model().model, DEFAULT_VLLM_MODEL)
        self.assertEqual(default_model_name("Qwen/Qwen2.5-14B-Instruct"), "Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(default_local_model("Qwen/Qwen2.5-14B-Instruct").model, "Qwen/Qwen2.5-14B-Instruct")

    def test_cli_without_injected_agent_ensures_ollama_before_chat(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            config = ConfigStore(Path(tmpdir) / "config.json")
            fake_model = FakeModel([ChatResponse("ready", [])])
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with patch("builtins.input", side_effect=["hello", "quit"]), redirect_stdout(stdout):
                    exit_code = cli_main([], store=store, config=config)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with(None, None)
        self.assertIn("ready", stdout.getvalue())

    def test_cli_model_option_selects_ollama_model(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            fake_model = FakeModel([ChatResponse("ready", [])])
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with patch("builtins.input", side_effect=["hello", "quit"]), redirect_stdout(stdout):
                    exit_code = cli_main(["--backend", "vllm", "--model", "Qwen/Qwen2.5-14B-Instruct"], store=store)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with("Qwen/Qwen2.5-14B-Instruct", "vllm")
        self.assertIn("ready", stdout.getvalue())

    def test_cli_default_chat_records_session(self) -> None:
        agent = Agent(model=FakeModel([ChatResponse("offline answer", [])]))
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("builtins.input", side_effect=["hello", "quit"]), redirect_stdout(stdout):
                exit_code = cli_main([], agent=agent, store=store)

            sessions = store.list()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].message_count, 2)
        self.assertEqual(sessions[0].title, "hello")
        self.assertIn("offline answer", stdout.getvalue())

    def test_cli_exit_function_prompt_exits_without_model_call(self) -> None:
        agent = Agent(model=FakeModel([]))
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("builtins.input", side_effect=["exit()"]), redirect_stdout(stdout):
                exit_code = cli_main([], agent=agent, store=store)
            sessions = store.list()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].message_count, 0)
        self.assertTrue(is_exit_prompt("quit()"))

    def test_cli_shows_thinking_but_saves_clean_answer(self) -> None:
        raw = '<|channel>thought\nWorking through it.\n<channel|>Final answer.'
        agent = Agent(model=FakeModel([ChatResponse(raw, [])]))
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("builtins.input", side_effect=["hello", "quit"]), redirect_stdout(stdout):
                exit_code = cli_main([], agent=agent, store=store)
            session_id = store.latest_id()
            self.assertIsNotNone(session_id)
            messages = store.messages(session_id or "")

        self.assertEqual(exit_code, 0)
        self.assertIn("thinking:", stdout.getvalue())
        self.assertIn("Working through it.", stdout.getvalue())
        self.assertIn("Final answer.", stdout.getvalue())
        self.assertEqual(messages[-1]["content"], "Final answer.")
        self.assertNotIn("Working through it.", messages[-1]["content"])

    def test_typewriter_print_writes_character_by_character(self) -> None:
        writes: list[str] = []
        flushes = 0

        def flush() -> None:
            nonlocal flushes
            flushes += 1

        typewriter_print("abc", delay=0, write=writes.append, flush=flush)

        self.assertEqual(writes, ["a", "b", "c", "\n"])
        self.assertEqual(flushes, 4)

    def test_print_thinking_uses_typewriter_output(self) -> None:
        writes: list[str] = []

        with patch("zeno.cli.time.sleep") as sleep:
            with patch("zeno.cli.sys.stdout.write", side_effect=writes.append), patch("zeno.cli.sys.stdout.flush"):
                print_thinking("step one")

        self.assertEqual("".join(writes), "thinking:\n  step one\n")
        self.assertGreater(sleep.call_count, 0)

    def test_compact_messages_summarizes_old_history(self) -> None:
        messages = []
        for index in range(26):
            role = "user" if index % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"message {index}"})

        compacted = compact_messages(messages)

        self.assertEqual(len(compacted), 13)
        self.assertEqual(compacted[0]["role"], "assistant")
        self.assertIn("Earlier conversation summary", compacted[0]["content"])
        self.assertIn("message 0", compacted[0]["content"])
        self.assertEqual(compacted[1:], messages[-12:])

    def test_cli_help_exits_cleanly(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            cli_main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("usage:", stdout.getvalue())
        self.assertIn("task", stdout.getvalue())

    def test_cli_unknown_command_fails_without_traceback(self) -> None:
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            cli_main(["unknown"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("invalid choice", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_chat_runtime_error_is_user_facing(self) -> None:
        class BrokenModel:
            def chat(
                self,
                messages: list[Message],
                tools: list[dict[str, object]] | None = None,
            ) -> ChatResponse:
                raise RuntimeError("Could not reach local model endpoint: http://localhost:11434/api/chat")

        stderr = io.StringIO()
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("builtins.input", side_effect=["hello"]), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli_main([], agent=Agent(model=BrokenModel()), store=store)

        self.assertEqual(exit_code, 1)
        self.assertIn("error: Could not reach local model endpoint", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_vllm_backend_error_is_user_facing(self) -> None:
        stderr = io.StringIO()

        with patch("zeno.cli.ensure_default_local_model", side_effect=RuntimeError("vllm command not found")):
            with redirect_stderr(stderr):
                exit_code = cli_main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("error: vllm command not found", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_chat_can_exit_without_human_input(self) -> None:
        agent = Agent(model=FakeModel([]))
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("builtins.input", side_effect=["quit"]), redirect_stdout(stdout):
                self.assertEqual(cli_main([], agent=agent, store=store), 0)

            sessions = store.list()

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].message_count, 0)

    def test_cli_task_list_prints_tasks(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            session_id = store.create()
            store.append(session_id, "user", "hello")
            store.append(session_id, "assistant", "hi")

            with redirect_stdout(stdout):
                exit_code = cli_main(["task", "list"], store=store)

        self.assertEqual(exit_code, 0)
        self.assertIn("ID\tUPDATED\tMESSAGES\tTITLE", stdout.getvalue())
        self.assertIn("hello", stdout.getvalue())
        self.assertIn("\t2\t", stdout.getvalue())

    def test_cli_task_list_does_not_bootstrap_ollama(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("zeno.cli.ensure_default_local_model", side_effect=RuntimeError("should not run")):
                with redirect_stdout(stdout):
                    exit_code = cli_main(["task", "list"], store=store)

        self.assertEqual(exit_code, 0)
        self.assertIn("No tasks found.", stdout.getvalue())

    def test_cli_task_list_ignores_model_option(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with patch("zeno.cli.ensure_default_local_model", side_effect=RuntimeError("should not run")):
                with redirect_stdout(stdout):
                    exit_code = cli_main(["--backend", "vllm", "--model", "Qwen/Qwen2.5-14B-Instruct", "task", "list"], store=store)

        self.assertEqual(exit_code, 0)
        self.assertIn("No tasks found.", stdout.getvalue())

    def test_cli_session_command_is_removed(self) -> None:
        stderr = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            cli_main(["session", "list"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_cli_task_create_runs_once_and_records_session(self) -> None:
        stdout = io.StringIO()
        agent = Agent(model=FakeModel([ChatResponse("task done", [])]))

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with redirect_stdout(stdout):
                exit_code = cli_main(["task", "create", "write an mlp trainer"], agent=agent, store=store)

            sessions = store.list()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "task done")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].message_count, 2)
        self.assertEqual(sessions[0].title, "write an mlp trainer")

    def test_cli_task_create_uses_model_option(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            fake_model = FakeModel([ChatResponse("task done", [])])
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with redirect_stdout(stdout):
                    exit_code = cli_main(["--backend", "vllm", "--model", "Qwen/Qwen2.5-14B-Instruct", "task", "create", "write an mlp trainer"], store=store)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with("Qwen/Qwen2.5-14B-Instruct", "vllm")
        self.assertIn("task done", stdout.getvalue())

    def test_cli_default_agent_can_write_files_with_tool_call(self) -> None:
        model = FakeModel(
            [
                ChatResponse("", [ToolCall(name="write_file", arguments={"path": "train_mlp.py", "content": "print('mlp')\n"})]),
                ChatResponse("Created train_mlp.py", []),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                agent = default_agent(model)
                stdout = io.StringIO()
                store = SessionStore(Path(tmpdir) / ".zeno" / "sessions")
                with redirect_stdout(stdout):
                    exit_code = cli_main(["task", "create", "write an mlp trainer"], agent=agent, store=store)
                content = (Path(tmpdir) / "train_mlp.py").read_text(encoding="utf-8")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(exit_code, 0)
        self.assertEqual(content, "print('mlp')\n")
        self.assertIn("Created train_mlp.py", stdout.getvalue())
        self.assertIsNotNone(model.tools[0])

    def test_cli_default_agent_prompt_includes_inline_write_file_format(self) -> None:
        model = FakeModel([ChatResponse("done", [])])

        agent = default_agent(model)

        self.assertIn("call write_file", agent.system)
        self.assertIn("<|tool_call>call:write_file", agent.system)
        self.assertIn("Do not say you created a file unless you called write_file", agent.system)

    def test_cli_serve_starts_backend_without_chat(self) -> None:
        stdout = io.StringIO()
        fake_model = FakeModel([])
        fake_model.model = "Qwen/Qwen2.5-7B-Instruct"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigStore(Path(tmpdir) / "config.json")
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with redirect_stdout(stdout):
                    exit_code = cli_main(["--backend", "vllm", "serve"], config=config)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with(None, "vllm")
        self.assertIn("serving model: Qwen/Qwen2.5-7B-Instruct", stdout.getvalue())

    def test_cli_device_option_passes_backend_override(self) -> None:
        fake_model = FakeModel([])
        fake_model.model = "Qwen/Qwen2.5-7B-Instruct"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigStore(Path(tmpdir) / "config.json")
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with redirect_stdout(io.StringIO()):
                    exit_code = cli_main(["--backend", "vllm", "--device", "cpu", "serve"], config=config)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with(None, "vllm", device="cpu")

    def test_cli_startup_timeout_option_passes_backend_timeout(self) -> None:
        fake_model = FakeModel([])
        fake_model.model = "Qwen/Qwen2.5-7B-Instruct"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigStore(Path(tmpdir) / "config.json")
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with redirect_stdout(io.StringIO()):
                    exit_code = cli_main(["--backend", "vllm", "--startup-timeout", "3600", "serve"], config=config)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with(None, "vllm", startup_timeout=3600.0)

    def test_cli_reuses_saved_model_before_builtin_default(self) -> None:
        fake_model = FakeModel([])
        fake_model.model = "saved/model"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigStore(Path(tmpdir) / "config.json")
            config.save_model("vllm-mlx", "saved/model")
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model) as ensure_model:
                with redirect_stdout(io.StringIO()):
                    exit_code = cli_main(["--backend", "vllm-mlx", "serve"], config=config)

        self.assertEqual(exit_code, 0)
        ensure_model.assert_called_once_with("saved/model", "vllm-mlx")

    def test_cli_model_option_saves_explicit_choice(self) -> None:
        fake_model = FakeModel([])
        fake_model.model = "explicit/model"

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigStore(Path(tmpdir) / "config.json")
            with patch("zeno.cli.ensure_default_local_model", return_value=fake_model):
                with redirect_stdout(io.StringIO()):
                    exit_code = cli_main(["--backend", "vllm-mlx", "--model", "explicit/model", "serve"], config=config)
            saved_model = config.model_for_backend("vllm-mlx")

        self.assertEqual(exit_code, 0)
        self.assertEqual(saved_model, "explicit/model")

    def test_cli_task_create_error_is_user_facing(self) -> None:
        class BrokenModel:
            def chat(
                self,
                messages: list[Message],
                tools: list[dict[str, object]] | None = None,
            ) -> ChatResponse:
                raise RuntimeError("model failed")

        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with redirect_stderr(stderr):
                exit_code = cli_main(["task", "create", "write an mlp trainer"], agent=Agent(model=BrokenModel()), store=store)
            sessions = store.list()

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(sessions), 0)
        self.assertIn("error: model failed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_default_session_dir_is_workspace_local(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                self.assertEqual(default_session_dir(), Path(tmpdir) / ".zeno" / "sessions")
            finally:
                os.chdir(original_cwd)

    def test_cli_continue_without_sessions_is_user_facing(self) -> None:
        stderr = io.StringIO()
        agent = Agent(model=FakeModel([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            with redirect_stderr(stderr):
                exit_code = cli_main(["--continue"], agent=agent, store=store)

        self.assertEqual(exit_code, 1)
        self.assertIn("error: no task sessions found in this workspace", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_continue_reuses_latest_session_history(self) -> None:
        stdout = io.StringIO()
        model = FakeModel([ChatResponse("continued", [])])
        agent = Agent(model=model, system="system prompt")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            session_id = store.create()
            store.append(session_id, "user", "first question")
            store.append(session_id, "assistant", "first answer")

            with patch("builtins.input", side_effect=["follow up", "quit"]), redirect_stdout(stdout):
                exit_code = cli_main(["--continue"], agent=agent, store=store)

            sessions = store.list()

        self.assertEqual(exit_code, 0)
        self.assertEqual(sessions[0].message_count, 4)
        self.assertEqual(model.messages[0], [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ])
        self.assertIn(f"zeno task: {session_id}", stdout.getvalue())
        self.assertIn("continued", stdout.getvalue())

    def test_cli_compacts_long_session_history(self) -> None:
        model = FakeModel([ChatResponse("compacted", [])])
        agent = Agent(model=model, system="system prompt")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            session_id = store.create()
            for index in range(26):
                role = "user" if index % 2 == 0 else "assistant"
                store.append(session_id, role, f"message {index}")

            with patch("builtins.input", side_effect=["follow up", "quit"]), redirect_stdout(io.StringIO()):
                exit_code = cli_main(["--continue"], agent=agent, store=store)

        self.assertEqual(exit_code, 0)
        sent = model.messages[0]
        self.assertEqual(sent[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(sent[1]["role"], "assistant")
        self.assertIn("Earlier conversation summary", sent[1]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "follow up"})
        self.assertLess(len(sent), 30)


class OllamaManagerTests(unittest.TestCase):
    def test_ensure_ready_pulls_missing_model_when_service_running(self) -> None:
        manager = FakeOllamaManager(running=True, has_model=False)

        manager.ensure_ready()

        self.assertEqual(manager.calls, ["require_cli", "is_running", "has_model", "pull_model"])

    def test_ensure_ready_starts_service_before_pulling(self) -> None:
        manager = FakeOllamaManager(running=False, has_model=True)

        manager.ensure_ready()

        self.assertEqual(
            manager.calls,
            ["require_cli", "is_running", "start_service", "wait_until_running", "has_model"],
        )

    def test_has_model_accepts_latest_alias(self) -> None:
        manager = FakeOllamaManager(running=True, has_model=True)

        self.assertTrue(manager._has_model())

    def test_pull_model_uses_ollama_http_api(self) -> None:
        manager = FakeHttpPullOllamaManager()

        manager._pull_model()

        self.assertEqual(manager.pull_requests, [("/api/pull", {"name": "qwen3:14b", "stream": False}, None)])


class VllmFamilyManagerTests(unittest.TestCase):
    def test_vllm_mlx_command(self) -> None:
        manager = VllmFamilyManager(model="mlx-community/Qwen2.5-7B-Instruct-4bit", backend="vllm-mlx")

        self.assertEqual(manager._command(), ["vllm-mlx", "serve", "mlx-community/Qwen2.5-7B-Instruct-4bit", "--port", "8000"])
        self.assertEqual(manager.openai_base_url(), "http://localhost:8000/v1")

    def test_vllm_command(self) -> None:
        manager = VllmFamilyManager(model="Qwen/Qwen2.5-7B-Instruct", backend="vllm")

        self.assertEqual(manager._command(), ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct", "--port", "8000"])

    def test_vllm_command_accepts_device_override(self) -> None:
        manager = VllmFamilyManager(model="Qwen/Qwen2.5-0.5B-Instruct", backend="vllm", device="cpu")

        self.assertEqual(manager._command(), ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct", "--port", "8000", "--device", "cpu"])

    def test_default_startup_timeout_is_long_enough_for_downloads(self) -> None:
        manager = VllmFamilyManager(model="model", backend="vllm")

        self.assertGreaterEqual(manager.startup_timeout, 1800.0)

    def test_unsupported_backend_raises(self) -> None:
        manager = VllmFamilyManager(model="model", backend="bad")

        with self.assertRaisesRegex(RuntimeError, "Unsupported backend"):
            manager._command()

    def test_verbose_manager_logs_missing_command_check(self) -> None:
        messages: list[str] = []
        manager = VllmFamilyManager(model="model", backend="vllm", log=messages.append)

        with patch("zeno.vllm_family.shutil.which", return_value="/usr/bin/vllm"):
            manager._require_command("vllm")

        self.assertEqual(messages, ["found backend command: vllm"])

    def test_startup_failure_includes_backend_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VllmFamilyManager(model="model", backend="vllm", log_dir=Path(tmpdir))
            log_path = manager._backend_log_path()
            log_path.write_text("line one\nline two\n")

            message = manager._startup_failure_message("vllm service did not become ready")

        self.assertIn("vllm service did not become ready", message)
        self.assertIn("See backend log:", message)
        self.assertIn("line one", message)
        self.assertIn("line two", message)

    def test_wait_until_running_reports_process_exit(self) -> None:
        class ExitedProcess:
            returncode = 7

            def poll(self) -> int:
                return 7

        class NeverReadyManager(VllmFamilyManager):
            def _is_running(self) -> bool:
                return False

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = NeverReadyManager(model="model", backend="vllm", startup_timeout=0.1, log_dir=Path(tmpdir))
            manager._backend_log_path().write_text("backend crashed\n")

            with self.assertRaisesRegex(RuntimeError, "exited with code 7") as raised:
                manager._wait_until_running(ExitedProcess())

        self.assertIn("backend crashed", str(raised.exception))


class FakeOllamaManager(OllamaManager):
    def __init__(self, running: bool, has_model: bool) -> None:
        super().__init__(model="qwen3:14b")
        self.running = running
        self.has_model = has_model
        self.calls: list[str] = []

    def _require_cli(self) -> None:
        self.calls.append("require_cli")

    def _is_running(self) -> bool:
        self.calls.append("is_running")
        return self.running

    def _start_service(self) -> None:
        self.calls.append("start_service")

    def _wait_until_running(self) -> None:
        self.calls.append("wait_until_running")

    def _has_model(self) -> bool:
        self.calls.append("has_model")
        return self.has_model

    def _pull_model(self) -> None:
        self.calls.append("pull_model")


class FakeHttpPullOllamaManager(OllamaManager):
    def __init__(self) -> None:
        super().__init__(model="qwen3:14b")
        self.pull_requests: list[tuple[str, dict[str, object], float | None]] = []

    def _post_json(self, path: str, payload: dict[str, object], timeout: float | None) -> dict[str, object]:
        self.pull_requests.append((path, payload, timeout))
        return {"status": "success"}


if __name__ == "__main__":
    unittest.main()
