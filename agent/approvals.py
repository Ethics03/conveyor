from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from agent.models import ToolCall
from tools.base import ExecutionContext, Tool


PolicyAction = Literal["allow", "ask", "deny"]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallDecision:
    tool_call: ToolCall
    decision: PolicyDecision


class ApprovalPolicy(Protocol):
    def evaluate(
        self,
        *,
        tool: Tool,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> PolicyDecision: ...

class DefaultApprovalPolicy:
    def evaluate(
        self,
        *,
        tool: Tool,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> PolicyDecision:
        if tool.permission == "read":
            return PolicyDecision(action="allow")

        reason = (
            f"{tool.name} can modify workspace files"
            if tool.permission == "write"
            else f"{tool.name} can perform dangerous operations"
        )
        return PolicyDecision(
            action="ask",
            reason=reason,
        )
