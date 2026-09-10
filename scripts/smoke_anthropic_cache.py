from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent.models import Agent, Message, RunOutcome
from agent.runtime import Runtime
from providers.anthropic_provider import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider
from storage.store import Store
from tools.registry import ToolRegistry

DEFAULT_HISTORY_WORDS = 12_000


def _final_message(outcome: RunOutcome) -> Message:
    if outcome.final_message is None:
        raise RuntimeError("Run returned no final message")
    return outcome.final_message


def _usage(message: Message) -> dict[str, object]:
    usage = message.metadata.get("usage")
    if not isinstance(usage, dict):
        raise TypeError("Anthropic response did not include usage metadata")
    return dict(usage)


def _token_count(usage: dict[str, object], name: str) -> int:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live-test Anthropic automatic prompt caching across two turns. "
            "This performs paid API calls."
        )
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="confirm that paid Anthropic requests are intended",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CONVEYOR_MODEL", DEFAULT_ANTHROPIC_MODEL),
        help="Claude model used for the live requests",
    )
    parser.add_argument(
        "--history-words",
        type=int,
        default=DEFAULT_HISTORY_WORDS,
        help="size of the stable transcript prefix",
    )
    args = parser.parse_args()

    if not args.confirm_cost:
        parser.error("--confirm-cost is required because this performs paid API calls")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is required in the environment or .env")
    if args.history_words < 4_096:
        parser.error("--history-words must be at least 4096 to exceed cache thresholds")

    descriptor, database_path = tempfile.mkstemp(
        prefix="conveyor-anthropic-cache-",
        suffix=".db",
    )
    os.close(descriptor)
    database = Path(database_path)

    try:
        provider = AnthropicProvider(
            model=args.model,
            prompt_caching=True,
            native_compaction=False,
        )
        with Runtime(
            store=Store(database),
            provider=provider,
            registry=ToolRegistry(),
            workspace=Path.cwd(),
        ) as runtime:
            session = runtime.create_session("Anthropic prompt caching smoke")
            runtime.store.save_message(
                Message(
                    session_id=session.id,
                    role="user",
                    content="Stable project context: "
                    + "context " * args.history_words,
                )
            )
            runtime.store.save_message(
                Message(
                    session_id=session.id,
                    role="assistant",
                    content="I have loaded the stable project context.",
                )
            )
            agent = Agent(
                name="Prompt cache smoke agent",
                instructions="Follow the latest user instruction exactly.",
                model=args.model,
            )

            warm_outcome = runtime.run_turn(
                agent=agent,
                session=session,
                content="Reply with exactly CACHE_WARM.",
            )
            hit_outcome = runtime.run_turn(
                agent=agent,
                session=session,
                content="Reply with exactly CACHE_HIT.",
            )

        warm_message = _final_message(warm_outcome)
        hit_message = _final_message(hit_outcome)
        warm_usage = _usage(warm_message)
        hit_usage = _usage(hit_message)
        created_tokens = _token_count(warm_usage, "cache_creation_input_tokens")
        read_tokens = _token_count(hit_usage, "cache_read_input_tokens")
        if created_tokens <= 0:
            raise RuntimeError("First turn did not create an Anthropic prompt cache")
        if read_tokens <= 0:
            raise RuntimeError(
                "Second turn did not read from the Anthropic prompt cache"
            )

        print(
            json.dumps(
                {
                    "model": args.model,
                    "history_words": args.history_words,
                    "warm_turn": {
                        "status": warm_outcome.run.status,
                        "response": warm_message.content,
                        "usage": warm_usage,
                    },
                    "hit_turn": {
                        "status": hit_outcome.run.status,
                        "response": hit_message.content,
                        "usage": hit_usage,
                    },
                    "cache": {
                        "created_tokens": created_tokens,
                        "read_tokens": read_tokens,
                        "hit": True,
                    },
                },
                indent=2,
            )
        )
    finally:
        for candidate in (
            database,
            Path(f"{database}-shm"),
            Path(f"{database}-wal"),
        ):
            candidate.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
