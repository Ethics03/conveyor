from __future__ import annotations

import argparse
import json

from agent.context import (
    MIN_CLEARABLE_TOOL_RESULT_CHARS,
    clear_tool_results,
    group_conversation_turns,
)
from agent.models import Message, ToolCall


def _tool_turn(
    *,
    prompt: str,
    call_id: str,
    payload: str,
) -> tuple[list[Message], Message]:
    tool_call = ToolCall(
        id=call_id,
        name="read_file",
        arguments={"path": f"{call_id}.txt"},
    )
    result = Message(
        role="tool",
        name=tool_call.name,
        tool_call_id=tool_call.id,
        content=payload,
    )
    return (
        [
            Message(role="user", content=prompt),
            Message(role="assistant", tool_calls=[tool_call]),
            result,
            Message(role="assistant", content="Read complete."),
        ],
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test deterministic tool-result clearing",
    )
    parser.add_argument(
        "--payload-chars",
        type=int,
        default=MIN_CLEARABLE_TOOL_RESULT_CHARS * 2,
        help="number of characters in each synthetic tool result",
    )
    args = parser.parse_args()
    if args.payload_chars < MIN_CLEARABLE_TOOL_RESULT_CHARS:
        parser.error(
            "--payload-chars must be at least "
            f"{MIN_CLEARABLE_TOOL_RESULT_CHARS}"
        )

    old_turn, old_result = _tool_turn(
        prompt="Read the old file.",
        call_id="call_old",
        payload="o" * args.payload_chars,
    )
    recent_turn, recent_result = _tool_turn(
        prompt="Read the recent file.",
        call_id="call_recent",
        payload="r" * args.payload_chars,
    )
    current_turn = [Message(role="user", content="Summarize the work.")]
    transcript = [*old_turn, *recent_turn, *current_turn]

    reduced = clear_tool_results(transcript)
    reduced_old = next(
        message for message in reduced if message.tool_call_id == "call_old"
    )
    reduced_recent = next(
        message for message in reduced if message.tool_call_id == "call_recent"
    )
    old_assistant = next(
        message
        for message in reduced
        if any(tool_call.id == "call_old" for tool_call in message.tool_calls)
    )
    assistant_call_ids = {
        tool_call.id
        for message in reduced
        for tool_call in message.tool_calls
    }
    result_call_ids = {
        message.tool_call_id
        for message in reduced
        if message.role == "tool"
    }

    original_unchanged = old_result.content == "o" * args.payload_chars
    old_result_cleared = reduced_old.content != old_result.content
    recent_result_preserved = reduced_recent.content == recent_result.content
    pairing_preserved = assistant_call_ids == result_call_ids
    order_preserved = [message.id for message in reduced] == [
        message.id for message in transcript
    ]

    assert original_unchanged
    assert old_result_cleared
    assert recent_result_preserved
    assert pairing_preserved
    assert order_preserved

    print(
        json.dumps(
            {
                "turn_count": len(group_conversation_turns(transcript)),
                "message_count": len(transcript),
                "payload_chars": args.payload_chars,
                "old_result": {
                    "stored_chars": len(old_result.content),
                    "active_chars": len(reduced_old.content),
                    "cleared": old_result_cleared,
                },
                "recent_result": {
                    "stored_chars": len(recent_result.content),
                    "active_chars": len(reduced_recent.content),
                    "preserved": recent_result_preserved,
                },
                "original_unchanged": original_unchanged,
                "pairing_preserved": pairing_preserved,
                "order_preserved": order_preserved,
                "compacted_context_format": [
                    {
                        "role": old_assistant.role,
                        "content": old_assistant.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            }
                            for tool_call in old_assistant.tool_calls
                        ],
                    },
                    {
                        "role": reduced_old.role,
                        "name": reduced_old.name,
                        "tool_call_id": reduced_old.tool_call_id,
                        "content": reduced_old.content,
                    },
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
