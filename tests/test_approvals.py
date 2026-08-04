from __future__ import annotations

from agent.approvals import DefaultApprovalPolicy
from agent.models import ToolCall
from tools.base import ExecutionContext, tool


def test_default_policy_allows_read_tools(tmp_path) -> None:
    @tool(permission="read")
    def inspect_workspace() -> str:
        return ""

    decision = DefaultApprovalPolicy().evaluate(
        tool=inspect_workspace,
        tool_call=ToolCall(name=inspect_workspace.name),
        context=ExecutionContext(workspace=tmp_path),
    )

    assert decision.action == "allow"
    assert decision.reason == ""


def test_default_policy_asks_for_write_tools(tmp_path) -> None:
    @tool(permission="write")
    def update_workspace() -> str:
        return ""

    decision = DefaultApprovalPolicy().evaluate(
        tool=update_workspace,
        tool_call=ToolCall(name=update_workspace.name),
        context=ExecutionContext(workspace=tmp_path),
    )

    assert decision.action == "ask"
    assert decision.reason == "update_workspace can modify workspace files"


def test_default_policy_asks_for_dangerous_tools(tmp_path) -> None:
    @tool(permission="dangerous")
    def run_command() -> str:
        return ""

    decision = DefaultApprovalPolicy().evaluate(
        tool=run_command,
        tool_call=ToolCall(name=run_command.name),
        context=ExecutionContext(workspace=tmp_path),
    )

    assert decision.action == "ask"
    assert decision.reason == "run_command can perform dangerous operations"
