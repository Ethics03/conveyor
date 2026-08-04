from __future__ import annotations

from agent.approvals import ApprovalPolicy, PolicyDecision, ToolCallDecision
from agent.context import build_provider_messages
from agent.models import (
    Agent,
    Event,
    Message,
    ProviderResponse,
    Run,
    RunOutcome,
    Session,
    ToolCall,
    utc_now,
)
from providers import Provider
from providers.base import ProviderRequest
from storage.store import Store
from tools.base import ExecutionContext
from tools.registry import ToolRegistry


MAX_ITERATIONS = 20


def _start_run(
    *,
    agent: Agent,
    session: Session,
    store: Store,
    parent_run_id: str | None = None,
) -> Run:
    run = Run(
        session_id=session.id,
        agent_id=agent.id,
        parent_run_id=parent_run_id,
        status="running",
    )

    store.save_run(run)
    store.append_event(
        Event(
            type="run.started",
            session_id=session.id,
            run_id=run.id,
            payload={"agent_id": agent.id},
        )
    )

    return run


def _build_provider_request(
    *,
    agent: Agent,
    messages: list[Message],
    registry: ToolRegistry,
) -> ProviderRequest:
    return ProviderRequest(
        messages=build_provider_messages(agent, messages),
        tools=registry.schemas(),
        model=agent.model,
    )


def _save_assistant_message(
    *,
    response: ProviderResponse,
    session: Session,
    run: Run,
    store: Store,
) -> Message:
    metadata = dict(response.raw)
    if response.finish_reason is not None:
        metadata["finish_reason"] = response.finish_reason

    message = Message(
        session_id=session.id,
        run_id=run.id,
        role="assistant",
        content=response.content,
        tool_calls=list(response.tool_calls),
        metadata=metadata,
    )

    store.save_message(message)
    store.append_event(
        Event(
            type="message.created",
            session_id=session.id,
            run_id=run.id,
            message_id=message.id,
            payload={
                "role": message.role,
                "tool_call_count": len(message.tool_calls),
            },
        )
    )
    return message


def _preflight_tool_calls(
    *,
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    context: ExecutionContext,
    policy: ApprovalPolicy,
) -> list[ToolCallDecision]:
    decisions: list[ToolCallDecision] = []

    for tool_call in tool_calls:
        tool = registry.get(tool_call.name)

        if tool is None:
            decision = PolicyDecision(
                action="deny",
                reason=f"Unknown tool: {tool_call.name}",
            )
        else:
            decision = policy.evaluate(
                tool=tool,
                tool_call=tool_call,
                context=context,
            )

        decisions.append(
            ToolCallDecision(
                tool_call=tool_call,
                decision=decision,
            )
        )

    return decisions


def _execute_tool_call(
    *,
    tool_call: ToolCall,
    session: Session,
    run: Run,
    registry: ToolRegistry,
    context: ExecutionContext,
    store: Store,
) -> Message:
    store.append_event(
        Event(
            type="tool.started",
            session_id=session.id,
            run_id=run.id,
            payload={
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        )
    )

    result = registry.execute(tool_call, context)

    metadata = dict(result.metadata)
    metadata["ok"] = result.ok

    message = Message(
        session_id=session.id,
        run_id=run.id,
        role="tool",
        content=result.content,
        name=result.name,
        tool_call_id=result.tool_call_id,
        metadata=metadata,
    )

    store.save_message(message)
    store.append_event(
        Event(
            type="message.created",
            session_id=session.id,
            run_id=run.id,
            message_id=message.id,
            payload={
                "role": message.role,
                "name": result.name,
                "ok": result.ok,
            },
        )
    )
    store.append_event(
        Event(
            type="tool.finished",
            session_id=session.id,
            run_id=run.id,
            message_id=message.id,
            payload={
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "ok": result.ok,
            },
        )
    )

    return message


def _finish_run(
    *,
    run: Run,
    final_message: Message,
    iterations: int,
    store: Store,
) -> RunOutcome:
    run.status = "finished"
    run.error = None
    run.updated_at = utc_now()

    store.save_run(run)
    store.append_event(
        Event(
            type="run.finished",
            session_id=run.session_id,
            run_id=run.id,
            message_id=final_message.id,
            payload={"iterations": iterations},
        )
    )

    return RunOutcome(
        run=run,
        final_message=final_message,
        iterations=iterations,
    )


def _fail_run(
    *,
    run: Run,
    error: str,
    iterations: int,
    store: Store,
    final_message: Message | None = None,
) -> RunOutcome:
    run.status = "failed"
    run.error = error
    run.updated_at = utc_now()

    store.save_run(run)
    store.append_event(
        Event(
            type="run.failed",
            session_id=run.session_id,
            run_id=run.id,
            message_id=final_message.id if final_message else None,
            payload={
                "error": error,
                "iterations": iterations,
            },
        )
    )

    return RunOutcome(
        run=run,
        final_message=final_message,
        iterations=iterations,
    )


def run_agent(
    *,
    agent: Agent,
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    context: ExecutionContext,
    store: Store,
    parent_run_id: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> RunOutcome:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    allowed_registry = registry.subset(agent.tools)
    messages = store.list_messages(session.id)
    if not messages:
        raise ValueError("Cannot run an agent without session messages")

    run = _start_run(
        agent=agent,
        session=session,
        store=store,
        parent_run_id=parent_run_id,
    )
    final_message: Message | None = None

    iterations = 0

    try:
        for iterations in range(1, max_iterations + 1):
            request = _build_provider_request(
                agent=agent,
                messages=messages,
                registry=allowed_registry,
            )
            response = provider.generate(request)

            final_message = _save_assistant_message(
                response=response,
                session=session,
                run=run,
                store=store,
            )
            messages.append(final_message)

            if not final_message.tool_calls:
                return _finish_run(
                    run=run,
                    final_message=final_message,
                    iterations=iterations,
                    store=store,
                )

            for tool_call in final_message.tool_calls:
                tool_message = _execute_tool_call(
                    tool_call=tool_call,
                    session=session,
                    run=run,
                    registry=allowed_registry,
                    context=context,
                    store=store,
                )
                messages.append(tool_message)

    except Exception as exc:
        _ = _fail_run(
            run=run,
            error=str(exc),
            iterations=iterations,
            store=store,
            final_message=final_message,
        )
        raise

    return _fail_run(
        run=run,
        error=f"Maximum iterations exceeded: {max_iterations}",
        iterations=iterations,
        store=store,
        final_message=final_message,
    )
