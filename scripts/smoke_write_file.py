from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

from agent.approvals import ApprovalCallback
from agent.loop import run_agent
from agent.models import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    Message,
    ProviderResponse,
    Session,
)
from providers.fake import FakeProvider
from storage.store import Store
from tools.base import ExecutionContext
from tools.registry import ToolRegistry
from tools.workspace import write_file


def _prompt_for_approval(approval: ApprovalRequest) -> ApprovalDecision:
    print(f"approval requested: {approval.tool_call.name}")
    print(json.dumps(approval.tool_call.arguments, indent=2))

    while True:
        try:
            answer = input("allow this write? [y/N] ").strip().lower()
        except EOFError:
            return "denied"
        if answer in {"y", "yes"}:
            return "approved"
        if answer in {"", "n", "no"}:
            return "denied"
        print("answer with y or n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test write_file through the agent approval pipeline"
    )
    parser.add_argument(
        "--decision",
        choices=("approved", "denied"),
        help="skip the prompt and use this approval decision",
    )
    args = parser.parse_args()
    decision = cast(ApprovalDecision | None, args.decision)

    path = "output/smoke.txt"
    content = "Conveyor write_file smoke test\n"

    with tempfile.TemporaryDirectory(prefix="conveyor-write-") as temporary:
        workspace = Path(temporary)
        target = workspace / path
        store = Store(":memory:")
        session = Session(title="write_file smoke test")
        store.save_session(session)
        store.save_message(
            Message(
                session_id=session.id,
                role="user",
                content="Write the smoke-test file.",
            )
        )

        provider = FakeProvider([
            ProviderResponse.tool(
                "write_file",
                {"path": path, "content": content},
                tool_call_id="call_smoke_write_file",
            ),
            ProviderResponse.message("Write request handled."),
        ])

        callback: ApprovalCallback
        if decision is None:
            callback = _prompt_for_approval
        else:
            callback = lambda _: decision

        try:
            outcome = run_agent(
                agent=Agent(tools=["write_file"]),
                session=session,
                provider=provider,
                registry=ToolRegistry([write_file]),
                context=ExecutionContext(workspace=workspace),
                store=store,
                approval_callback=callback,
            )

            approval = store.list_approvals(run_id=outcome.run.id)[0]
            file_exists = target.is_file()
            file_content = target.read_text(encoding="utf-8") if file_exists else None

            if approval.status == "approved":
                assert file_content == content
            else:
                assert file_exists is False

            print(
                json.dumps(
                    {
                        "workspace": str(workspace),
                        "run_status": outcome.run.status,
                        "approval_status": approval.status,
                        "file": {
                            "path": path,
                            "exists": file_exists,
                            "content": file_content,
                        },
                        "events": [
                            event.type
                            for event in store.list_events(run_id=outcome.run.id)
                        ],
                    },
                    indent=2,
                )
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
