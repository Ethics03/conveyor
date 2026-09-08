from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.context import MIN_CLEARABLE_TOOL_RESULT_CHARS
from agent.models import Agent, Message, ProviderResponse, ToolCall
from agent.runtime import Runtime
from providers.base import ModelLimits
from providers.fake import FakeProvider
from storage.store import Store
from tools.registry import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test context planning at the agent loop boundary",
    )
    parser.add_argument(
        "--payload-chars",
        type=int,
        default=MIN_CLEARABLE_TOOL_RESULT_CHARS * 10,
        help="size of the synthetic old tool result",
    )
    args = parser.parse_args()
    if args.payload_chars < MIN_CLEARABLE_TOOL_RESULT_CHARS:
        parser.error(
            f"--payload-chars must be at least {MIN_CLEARABLE_TOOL_RESULT_CHARS}"
        )

    store = Store()
    provider = FakeProvider(
        [ProviderResponse.message("Compacted context accepted.")],
        model_limits=ModelLimits(12_000, 1_000),
    )
    with Runtime(store, provider, ToolRegistry(), workspace=Path.cwd()) as runtime:
        session = runtime.create_session("Context integration smoke")
        old_call = ToolCall(id="call_old", name="read_file")
        old_result = Message(
            session_id=session.id,
            role="tool",
            name="read_file",
            tool_call_id=old_call.id,
            content="x" * args.payload_chars,
        )
        history = [
            Message(session_id=session.id, role="user", content="Old turn"),
            Message(
                session_id=session.id,
                role="assistant",
                tool_calls=[old_call],
            ),
            old_result,
            Message(session_id=session.id, role="assistant", content="Old answer"),
            Message(session_id=session.id, role="user", content="Recent turn one"),
            Message(session_id=session.id, role="assistant", content="Recent answer"),
        ]
        for message in history:
            store.save_message(message)

        outcome = runtime.run_turn(
            agent=Agent(name="Context smoke agent"),
            session=session,
            content="Continue from the current state.",
        )
        request = provider.requests[0]
        active_result = next(
            message
            for message in request.messages
            if message.tool_call_id == old_call.id
        )
        persisted_result = next(
            message
            for message in store.list_messages(session.id)
            if message.id == old_result.id
        )
        compaction_event = next(
            event
            for event in store.list_events(run_id=outcome.run.id)
            if event.type == "context.compaction_planned"
        )

        before = compaction_event.payload["input_tokens_before"]
        after = compaction_event.payload["input_tokens_after"]
        assert isinstance(before, int)
        assert isinstance(after, int)
        assert after < before
        assert compaction_event.payload["cleared_tool_results"] == 1
        assert persisted_result.content == "x" * args.payload_chars
        assert active_result.content != persisted_result.content

        print(
            json.dumps(
                {
                    "run_status": outcome.run.status,
                    "model_limits": {
                        "context_window_tokens": 12_000,
                        "max_output_tokens": request.max_tokens,
                    },
                    "context_plan": compaction_event.payload,
                    "tool_result": {
                        "stored_chars": len(persisted_result.content),
                        "provider_chars": len(active_result.content),
                        "provider_content": active_result.content,
                    },
                    "original_preserved": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
