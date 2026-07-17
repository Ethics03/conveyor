from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.models import (
    Agent,
    ApprovalRequest,
    Event,
    Message,
    ProviderMessage,
    Run,
    Session,
    ToolCall,
    ToolResult,
    utc_now,
)
from providers.base import Provider, ProviderRequest
from storage.store import Store
from tools.base import ExecutionContext, ToolRegistry

MAX_ITERATIONS = 20
