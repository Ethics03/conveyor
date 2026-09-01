from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from threading import Barrier, Lock

from agent.loop import run_agent
from agent.models import Agent, Message, ProviderResponse, Session, ToolCall
from providers.fake import FakeProvider
from storage.store import Store
from tools.base import ExecutionContext, tool
from tools.registry import ToolRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test controlled parallel tool execution"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="seconds each synthetic tool should run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.delay <= 0:
        raise ValueError("--delay must be greater than zero")

    rendezvous = Barrier(2)
    timings: dict[str, dict[str, float]] = {}
    timing_lock = Lock()

    def run_synthetic_read(name: str) -> str:
        started_at = time.monotonic()
        with timing_lock:
            timings[name] = {"started_at": started_at}

        _ = rendezvous.wait(timeout=max(2.0, args.delay * 4))
        time.sleep(args.delay)

        finished_at = time.monotonic()
        with timing_lock:
            timings[name]["finished_at"] = finished_at
        return name

    @tool(permission="read", parallel_safe=True)
    def read_alpha() -> str:
        return run_synthetic_read("read_alpha")

    @tool(permission="read", parallel_safe=True)
    def read_beta() -> str:
        return run_synthetic_read("read_beta")

    store = Store(":memory:")
    session = Session(title="parallel tool smoke test")
    store.save_session(session)
    store.save_message(
        Message(
            session_id=session.id,
            role="user",
            content="Run both independent reads.",
        )
    )

    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall(id="call_alpha", name="read_alpha"),
                    ToolCall(id="call_beta", name="read_beta"),
                ]
            ),
            ProviderResponse.message("Both reads completed."),
        ]
    )

    started_at = time.monotonic()
    outcome = run_agent(
        agent=Agent(tools=["read_alpha", "read_beta"]),
        session=session,
        provider=provider,
        registry=ToolRegistry([read_alpha, read_beta]),
        context=ExecutionContext(workspace=Path.cwd()),
        store=store,
    )
    elapsed = time.monotonic() - started_at

    tool_messages = [
        message
        for message in store.list_messages(session.id)
        if message.role == "tool"
    ]
    starts = [timings[name]["started_at"] for name in ["read_alpha", "read_beta"]]
    finishes = [
        timings[name]["finished_at"] for name in ["read_alpha", "read_beta"]
    ]
    overlap = max(starts) < min(finishes)

    if outcome.run.status != "finished":
        raise RuntimeError(f"agent run did not finish: {outcome.run.status}")
    if not overlap:
        raise RuntimeError("parallel-safe tool executions did not overlap")
    if [message.name for message in tool_messages] != ["read_alpha", "read_beta"]:
        raise RuntimeError("tool results were not preserved in model call order")
    if not all(message.metadata.get("ok") is True for message in tool_messages):
        raise RuntimeError("one or more parallel tool calls failed")

    print(
        json.dumps(
            {
                "run_status": outcome.run.status,
                "parallel": overlap,
                "delay_per_tool_seconds": args.delay,
                "elapsed_seconds": round(elapsed, 3),
                "sequential_minimum_seconds": round(args.delay * 2, 3),
                "result_order": [message.name for message in tool_messages],
                "results": [asdict(message) for message in tool_messages],
                "events": [
                    event.type for event in store.list_events(run_id=outcome.run.id)
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
