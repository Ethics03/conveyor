from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from agent.models import Agent, ProviderResponse, Session
from agent.runtime import Runtime
from providers.anthropic_provider import AnthropicProvider
from providers.base import Provider
from providers.fake import FakeProvider
from storage.store import Store
from tools.registry import ToolRegistry


def _database_path(value: str | None) -> tuple[Path, bool]:
    if value is not None:
        path = Path(value).expanduser().resolve()
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing database: {path}")
        return path, False

    descriptor, raw_path = tempfile.mkstemp(
        prefix="conveyor-session-titles-",
        suffix=".db",
    )
    os.close(descriptor)
    return Path(raw_path), True


def _remove_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-shm"), Path(f"{path}-wal")):
        candidate.unlink(missing_ok=True)


def _title_snapshot(session: Session) -> dict[str, str]:
    return {
        "title": session.title,
        "source": session.title_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test automatic and user session title precedence"
    )
    parser.add_argument(
        "database",
        nargs="?",
        help="optional new database path; temporary storage is used by default",
    )
    parser.add_argument(
        "--message",
        default="Investigate provider retries and timeout handling",
        help="first user message used to derive the automatic title",
    )
    parser.add_argument(
        "--user-title",
        default="Provider reliability work",
        help="explicit title that should take precedence over automatic titles",
    )
    parser.add_argument(
        "--generated-title",
        default="Provider Retry Investigation",
        help="simulated model title used when --live is not set",
    )
    parser.add_argument(
        "--model",
        help="optional model override",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use Anthropic for the agent response and title generation",
    )
    args = parser.parse_args()

    try:
        database, temporary = _database_path(args.database)
    except ValueError as exc:
        parser.error(str(exc))

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is required with --live")

    provider: Provider = (
        AnthropicProvider()
        if args.live
        else FakeProvider(
            [
                ProviderResponse.message("The first agent turn completed."),
                ProviderResponse.message(args.generated_title),
            ]
        )
    )

    try:
        with Runtime(
            store=Store(database),
            provider=provider,
            registry=ToolRegistry(),
            workspace=Path.cwd(),
        ) as runtime:
            session = runtime.create_session()
            initial = runtime.store.get_session(session.id)
            assert initial is not None

            outcome = runtime.run_turn(
                agent=Agent(
                    name="Session title smoke agent",
                    instructions="Answer the user concisely.",
                    model=args.model,
                ),
                session=session,
                content=args.message,
            )
            automatic = runtime.store.get_session(session.id)
            assert outcome.run.status == "finished"
            assert automatic is not None
            assert automatic.title_source == "auto"
            assert session == automatic

            user_updated = runtime.store.set_session_title(session.id, args.user_title)
            user = runtime.store.get_session(session.id)
            assert user_updated is True
            assert user is not None
            assert user.title_source == "user"

            late_auto_updated = runtime.store.set_auto_title(
                session.id,
                "This automatic title must not win",
            )
            final = runtime.store.get_session(session.id)
            assert late_auto_updated is False
            assert final is not None
            assert final.title == user.title
            assert final.title_source == "user"

        reopened_store = Store(database)
        try:
            reopened = reopened_store.get_session(session.id)
            assert reopened == final
        finally:
            reopened_store.close()

        print(
            json.dumps(
                {
                    "database": str(database),
                    "temporary": temporary,
                    "session_id": session.id,
                    "mode": "live" if args.live else "deterministic",
                    "message": args.message,
                    "assistant_response": outcome.final_message.content
                    if outcome.final_message is not None
                    else None,
                    "provider_calls": 2,
                    "transitions": {
                        "initial": _title_snapshot(initial),
                        "automatic": {
                            **_title_snapshot(automatic),
                        },
                        "user": {
                            "updated": user_updated,
                            **_title_snapshot(user),
                        },
                        "late_automatic": {
                            "updated": late_auto_updated,
                            **_title_snapshot(final),
                        },
                    },
                    "reopened": True,
                },
                indent=2,
            )
        )
    finally:
        if temporary:
            _remove_database(database)


if __name__ == "__main__":
    main()
