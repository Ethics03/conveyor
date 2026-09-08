from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent.models import Agent, Message, ProviderReplayState
from agent.runtime import Runtime
from providers.anthropic_provider import (
    DEFAULT_ANTHROPIC_COMPACTION_TRIGGER_TOKENS,
    DEFAULT_ANTHROPIC_MODEL,
    MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS,
    AnthropicProvider,
)
from storage.store import Store
from tools.registry import ToolRegistry

DEFAULT_HISTORY_WORDS = 65_000
DEFAULT_HISTORY_TURNS = 20


def _database_path(value: str | None) -> tuple[Path, bool]:
    if value is not None:
        path = Path(value).expanduser().resolve()
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing database: {path}")
        return path, False

    descriptor, raw_path = tempfile.mkstemp(
        prefix="conveyor-anthropic-compaction-",
        suffix=".db",
    )
    os.close(descriptor)
    return Path(raw_path), True


def _remove_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-shm"), Path(f"{path}-wal")):
        candidate.unlink(missing_ok=True)


def _seed_transcript(
    store: Store,
    *,
    session_id: str,
    history_words: int,
    turns: int,
) -> int:
    words_per_message, remainder = divmod(history_words, turns * 2)
    stored_chars = 0

    for index in range(turns * 2):
        word_count = words_per_message + (1 if index < remainder else 0)
        role = "user" if index % 2 == 0 else "assistant"
        content = (
            f"Historical turn {index + 1}. "
            + "context " * word_count
            + f"End of historical turn {index + 1}."
        )
        store.save_message(
            Message(
                session_id=session_id,
                role=role,
                content=content,
            )
        )
        stored_chars += len(content)

    return stored_chars


def _usage(message: Message | None) -> object:
    return message.metadata.get("usage") if message is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live-test Anthropic native compaction and durable checkpoint replay. "
            "This sends a large paid request."
        )
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="confirm that a paid Anthropic request of roughly 65k input tokens is intended",
    )
    parser.add_argument(
        "--database",
        help="optional new database path; temporary storage is used by default",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CONVEYOR_MODEL", DEFAULT_ANTHROPIC_MODEL),
        help="supported Claude model used for the live request",
    )
    parser.add_argument(
        "--trigger-tokens",
        type=int,
        default=MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS,
        help="native compaction threshold; Anthropic requires at least 50000",
    )
    parser.add_argument(
        "--history-words",
        type=int,
        default=DEFAULT_HISTORY_WORDS,
        help="number of synthetic one-token words distributed across the transcript",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="number of historical user/assistant turn pairs",
    )
    args = parser.parse_args()

    if not args.confirm_cost:
        parser.error("--confirm-cost is required because this performs paid API calls")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is required in the environment or .env")
    if args.trigger_tokens < MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS:
        parser.error(
            f"--trigger-tokens must be at least "
            f"{MIN_ANTHROPIC_COMPACTION_TRIGGER_TOKENS}"
        )
    if args.history_words <= args.trigger_tokens:
        parser.error("--history-words must be greater than --trigger-tokens")
    if args.turns < 1:
        parser.error("--turns must be at least 1")

    try:
        database, temporary = _database_path(args.database)
    except ValueError as exc:
        parser.error(str(exc))

    first_outcome = None
    second_outcome = None
    session_id = ""
    checkpoint: ProviderReplayState | None = None
    compaction_event_payload: dict[str, object] | None = None
    stored_chars = 0

    try:
        provider = AnthropicProvider(
            model=args.model,
            native_compaction=True,
            compaction_trigger_tokens=args.trigger_tokens,
        )
        with Runtime(
            store=Store(database),
            provider=provider,
            registry=ToolRegistry(),
            workspace=Path.cwd(),
        ) as runtime:
            session = runtime.create_session("Anthropic native compaction smoke")
            session_id = session.id
            stored_chars = _seed_transcript(
                runtime.store,
                session_id=session.id,
                history_words=args.history_words,
                turns=args.turns,
            )
            agent = Agent(
                name="Compaction smoke agent",
                instructions=(
                    "This is a compaction smoke test. Do not repeat the historical "
                    "transcript. Follow the latest user instruction exactly."
                ),
                model=args.model,
            )

            first_outcome = runtime.run_turn(
                agent=agent,
                session=session,
                content="Reply with exactly COMPACTION_OK.",
            )
            if first_outcome.final_message is None:
                raise RuntimeError("First turn returned no final message")

            checkpoint = ProviderReplayState.from_metadata(
                first_outcome.final_message.metadata.get("provider_replay_state")
            )
            if checkpoint is None:
                raise RuntimeError(
                    "Anthropic returned no compaction checkpoint. Increase "
                    "--history-words or verify that the selected model supports compaction."
                )

            compaction_event = next(
                (
                    event
                    for event in runtime.store.list_events(run_id=first_outcome.run.id)
                    if event.type == "context.compacted"
                ),
                None,
            )
            if compaction_event is None:
                raise RuntimeError("Compaction checkpoint was not recorded as an event")
            compaction_event_payload = dict(compaction_event.payload)

            second_outcome = runtime.run_turn(
                agent=agent,
                session=session,
                content="Reply with exactly REPLAY_OK.",
            )
            if second_outcome.final_message is None:
                raise RuntimeError("Replay turn returned no final message")

        reopened = Store(database)
        try:
            persisted_messages = reopened.list_messages(session_id)
            persisted_checkpoint = next(
                (
                    ProviderReplayState.from_metadata(
                        message.metadata.get("provider_replay_state")
                    )
                    for message in persisted_messages
                    if message.role == "assistant"
                    and "provider_replay_state" in message.metadata
                ),
                None,
            )
            if persisted_checkpoint != checkpoint:
                raise RuntimeError("Persisted compaction checkpoint did not round-trip")
        finally:
            reopened.close()

        assert first_outcome is not None
        assert first_outcome.final_message is not None
        assert second_outcome is not None
        assert second_outcome.final_message is not None
        print(
            json.dumps(
                {
                    "database": str(database),
                    "temporary": temporary,
                    "model": args.model,
                    "session_id": session_id,
                    "synthetic_history": {
                        "turns": args.turns,
                        "words": args.history_words,
                        "characters": stored_chars,
                    },
                    "native_compaction": {
                        "configured_trigger_tokens": args.trigger_tokens,
                        "default_trigger_tokens": (
                            DEFAULT_ANTHROPIC_COMPACTION_TRIGGER_TOKENS
                        ),
                        "checkpoint_items": len(checkpoint.items),
                        "event": compaction_event_payload,
                        "persisted": True,
                    },
                    "first_turn": {
                        "status": first_outcome.run.status,
                        "response": first_outcome.final_message.content,
                        "usage": _usage(first_outcome.final_message),
                    },
                    "replay_turn": {
                        "status": second_outcome.run.status,
                        "response": second_outcome.final_message.content,
                        "usage": _usage(second_outcome.final_message),
                    },
                },
                indent=2,
            )
        )
    finally:
        if temporary:
            _remove_database(database)


if __name__ == "__main__":
    main()
