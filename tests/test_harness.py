from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conveyor_harness.model import ModelDecision, ScriptedModelClient, ToolCall
from conveyor_harness.runtime import AgentSession
from conveyor_harness.tools.registry import ToolRegistry
from conveyor_harness.tools.workspace_tools import default_tools
from conveyor_harness.transcript import Transcript
from conveyor_harness.workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def test_blocks_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace.from_path(tmp)
            with self.assertRaises(WorkspaceError):
                workspace.resolve("../outside.txt")


class ToolTests(unittest.TestCase):
    def test_write_read_edit_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace.from_path(tmp)
            registry = ToolRegistry(default_tools())

            write = registry.execute(
                "write_file",
                workspace,
                {"path": "notes.txt", "content": "alpha\nbeta\n", "overwrite": False},
            )
            self.assertTrue(write["ok"])

            edit = registry.execute(
                "edit_file",
                workspace,
                {"path": "notes.txt", "old": "beta", "new": "gamma"},
            )
            self.assertTrue(edit["ok"])

            read = registry.execute("read_file", workspace, {"path": "notes.txt"})
            self.assertEqual(read["result"]["content"], "alpha\ngamma\n")

            search = registry.execute("search_text", workspace, {"query": "gamma"})
            self.assertEqual(search["result"]["matches"][0]["line"], 2)

    def test_run_command_blocks_destructive_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace.from_path(tmp)
            registry = ToolRegistry(default_tools())
            result = registry.execute(
                "run_command", workspace, {"command": "rm -rf anything"}
            )
            self.assertFalse(result["ok"])
            self.assertIn("requires explicit approval", result["error"])


class RuntimeTests(unittest.TestCase):
    def test_scripted_session_calls_tool_then_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript_path = Path(tmp) / "transcript.jsonl"
            model = ScriptedModelClient(
                [
                    ModelDecision(
                        tool_call=ToolCall(
                            name="write_file",
                            arguments={
                                "path": "hello.txt",
                                "content": "hello\n",
                                "overwrite": False,
                            },
                        )
                    ),
                    ModelDecision(final="done"),
                ]
            )
            session = AgentSession(
                workspace=Workspace.from_path(tmp),
                model=model,
                transcript=Transcript(transcript_path),
            )

            result = session.run("create hello.txt")

            self.assertEqual(result.final, "done")
            self.assertEqual((Path(tmp) / "hello.txt").read_text(encoding="utf-8"), "hello\n")
            events = [
                json.loads(line)["kind"]
                for line in transcript_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                events, ["session_started", "tool_call", "tool_result", "final"]
            )


if __name__ == "__main__":
    unittest.main()
