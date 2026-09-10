from __future__ import annotations

from dataclasses import dataclass, field

from anthropic import Anthropic, omit
from anthropic.types.beta import (
    BetaCompact20260112EditParam,
    BetaCompactionBlockParam,
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from agent.models import (
    ProviderMessage,
    ProviderReplayState,
    ProviderResponse,
    ToolCall,
)
from providers.base import (
    ModelLimits,
    ProviderRequest,
)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_CONTEXT_WINDOW_TOKENS = 200_000
DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_ANTHROPIC_COMPACTION_TRIGGER_TOKENS = 150_000
MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS = 50_000
COMPACTION_LOCAL_FALLBACK_MARGIN_TOKENS = 8_192
ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"

_COMPACTION_MODEL_MARKERS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
)


@dataclass(slots=True)
class AnthropicProvider:
    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key: str | None = field(default=None, repr=False)
    name: str = "anthropic"
    context_window_tokens: int = DEFAULT_ANTHROPIC_CONTEXT_WINDOW_TOKENS
    max_output_tokens: int = DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS
    prompt_caching: bool = True
    native_compaction: bool = True
    compaction_trigger_tokens: int = DEFAULT_ANTHROPIC_COMPACTION_TRIGGER_TOKENS
    _client: Anthropic = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.compaction_trigger_tokens < MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS:
            raise ValueError(
                "compaction_trigger_tokens must be at least "
                f"{MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS}"
            )
        self._client = Anthropic(api_key=self.api_key) if self.api_key else Anthropic()

    def close(self) -> None:
        self._client.close()

    def model_limits(self, model: str | None = None) -> ModelLimits:
        return ModelLimits(
            context_window_tokens=self.context_window_tokens,
            max_output_tokens=self.max_output_tokens,
        )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        model_name = request.model or self.model
        system = _system_instructions(request.messages)
        tools = _anthropic_tools(request)
        compaction_edit = _anthropic_compaction_edit(
            request,
            model=model_name,
            enabled=self.native_compaction,
            configured_trigger_tokens=self.compaction_trigger_tokens,
        )
        response = self._client.beta.messages.create(
            model=model_name,
            max_tokens=request.max_tokens or self.max_output_tokens,
            messages=_anthropic_messages(request.messages),
            temperature=(
                request.temperature if request.temperature is not None else omit
            ),
            system=system if system else omit,
            tools=tools if tools else omit,
            cache_control={"type": "ephemeral"} if self.prompt_caching else omit,
            betas=[ANTHROPIC_COMPACTION_BETA] if compaction_edit else omit,
            context_management=(
                {"edits": [compaction_edit]} if compaction_edit is not None else omit
            ),
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        compaction_blocks: list[dict[str, object]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )
            elif block.type == "compaction":
                compaction_block: dict[str, object] = {
                    "type": "compaction",
                    "content": block.content,
                }
                if block.encrypted_content is not None:
                    compaction_block["encrypted_content"] = block.encrypted_content
                compaction_blocks.append(compaction_block)

        replay_state = (
            ProviderReplayState(
                provider=self.name,
                items=tuple(compaction_blocks),
            )
            if compaction_blocks
            else None
        )

        return ProviderResponse(
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            raw={
                "provider": self.name,
                "model": response.model,
                "response_id": response.id,
                "usage": response.usage.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            },
            replay_state=replay_state,
        )


def _anthropic_messages(messages: list[ProviderMessage]) -> list[BetaMessageParam]:
    converted: list[BetaMessageParam] = []
    pending_tool_results: list[BetaToolResultBlockParam] = []

    for message in messages:
        if message.role == "system":
            continue

        if message.role == "tool":
            if message.tool_call_id is None:
                raise ValueError("Tool result message requires tool_call_id")
            tool_result: BetaToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            if message.is_error:
                tool_result["is_error"] = True
            pending_tool_results.append(tool_result)
            continue

        if pending_tool_results:
            converted.append(
                {
                    "role": "user",
                    "content": pending_tool_results,
                }
            )
            pending_tool_results = []

        role = "assistant" if message.role == "assistant" else "user"
        replay_blocks = _anthropic_replay_blocks(message)
        if message.role == "assistant" and (message.tool_calls or replay_blocks):
            content: list[BetaContentBlockParam] = list(replay_blocks)
            if message.content:
                text_block: BetaTextBlockParam = {
                    "type": "text",
                    "text": message.content,
                }
                content.append(text_block)
            for tool_call in message.tool_calls:
                tool_use_block: BetaToolUseBlockParam = {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
                content.append(tool_use_block)
            converted.append({"role": role, "content": content})
            continue

        converted.append({"role": role, "content": message.content})

    if pending_tool_results:
        converted.append(
            {
                "role": "user",
                "content": pending_tool_results,
            }
        )

    return converted or [{"role": "user", "content": ""}]


def _anthropic_tools(request: ProviderRequest) -> list[BetaToolParam]:
    converted: list[BetaToolParam] = []
    for tool in request.tools:
        input_schema: dict[str, object] = dict(tool.parameters)
        converted.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": input_schema
                or {
                    "type": "object",
                    "properties": {},
                },
            }
        )
    return converted


def _anthropic_replay_blocks(
    message: ProviderMessage,
) -> list[BetaCompactionBlockParam]:
    state = message.replay_state
    if state is None or state.provider != "anthropic":
        return []

    blocks: list[BetaCompactionBlockParam] = []
    for item in state.items:
        if item.get("type") != "compaction":
            continue
        content = item.get("content")
        encrypted_content = item.get("encrypted_content")
        if content is not None and not isinstance(content, str):
            continue
        if encrypted_content is not None and not isinstance(encrypted_content, str):
            continue
        block: BetaCompactionBlockParam = {
            "type": "compaction",
            "content": content,
        }
        if encrypted_content is not None:
            block["encrypted_content"] = encrypted_content
        blocks.append(block)
    return blocks


def _anthropic_compaction_edit(
    request: ProviderRequest,
    *,
    model: str,
    enabled: bool,
    configured_trigger_tokens: int,
) -> BetaCompact20260112EditParam | None:
    if not enabled or not _supports_native_compaction(model):
        return None

    context_plan = request.metadata.get("context_plan")
    if not isinstance(context_plan, dict):
        return None
    local_trigger = context_plan.get("trigger_tokens")
    if isinstance(local_trigger, bool) or not isinstance(local_trigger, int):
        return None

    native_trigger = min(
        configured_trigger_tokens,
        local_trigger - COMPACTION_LOCAL_FALLBACK_MARGIN_TOKENS,
    )
    if native_trigger < MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS:
        return None

    return {
        "type": "compact_20260112",
        "trigger": {
            "type": "input_tokens",
            "value": native_trigger,
        },
    }


def _supports_native_compaction(model: str) -> bool:
    normalized = model.lower()
    return any(marker in normalized for marker in _COMPACTION_MODEL_MARKERS)


def _system_instructions(messages: list[ProviderMessage]) -> str:
    instructions = [message.content for message in messages if message.role == "system"]
    return "\n\n".join(instructions)
