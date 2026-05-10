from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from zeno import Agent, ChatResponse, ConfigStore, DEFAULT_VLLM_MODEL, MLXChatModel, Message, OllamaChatModel, OllamaManager, OpenAICompatibleChatModel, SessionStore, ToolCall, VllmFamilyManager, default_backend, default_local_model, default_model_name, tool_schema
from zeno.models import _parse_tool_calls, MLXChatModel as ConcreteMLXChatModel
from zeno.cli import main as cli_main
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


class AgentTests(unittest.TestCase):
    def test_returns_plain_model_reply(self) -> None:
        agent = Agent(model=FakeModel([ChatResponse("hello", [])]), system="test")

        self.assertEqual(agent.run("hi"), "hello")

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
