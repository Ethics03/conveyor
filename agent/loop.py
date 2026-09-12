from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from agent.approvals import (
    ApprovalCallback,
    ApprovalPolicy,
    DefaultApprovalPolicy,
    PolicyDecision,
    ToolCallDecision,
)
from agent.context import ContextBudget, ContextPlan, plan_context
from agent.models import (
    Agent,
    ApprovalRequest,
    Event,
    Message,
    ProviderResponse,
    Run,
    RunOutcome,
    Session,
    ToolCall,
    ToolResult,
    utc_now,
)
from providers import Provider
from providers.base import ProviderRequest
from storage.store import Store
from tools.base import ExecutionContext
from tools.registry import ToolRegistry

MAX_ITERATIONS = 20
MAX_PARALLEL_TOOL_WORKERS = 8


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


def _plan_provider_request(
    *,
    agent: Agent,
    session: Session,
    run: Run,
    messages: list[Message],
    registry: ToolRegistry,
    provider: Provider,
    store: Store,
    iteration: int,
) -> ProviderRequest:
    limits = provider.model_limits(agent.model)
    budget = ContextBudget(
        context_window_tokens=limits.context_window_tokens,
        max_output_tokens=limits.max_output_tokens,
    )
    plan = plan_context(
        agent=agent,
        messages=messages,
        tools=registry.schemas(),
        budget=budget,
        session=session,
        run=run,
    )
    telemetry = _context_plan_telemetry(plan, budget=budget, iteration=iteration)
    plan.request.metadata["context_plan"] = telemetry

    if plan.compaction_triggered:
        store.append_event(
            Event(
                type="context.compaction_planned",
                session_id=session.id,
                run_id=run.id,
                payload=telemetry,
            )
        )
    return plan.request


def _context_plan_telemetry(
    plan: ContextPlan,
    *,
    budget: ContextBudget,
    iteration: int,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "input_tokens_before": plan.before.input_tokens,
        "input_tokens_after": plan.after.input_tokens,
        "token_count_source": plan.after.source,
        "trigger_tokens": budget.trigger_tokens,
        "target_tokens": budget.target_tokens,
        "compaction_triggered": plan.compaction_triggered,
        "cleared_tool_results": plan.cleared_tool_results,
        "summary_required": plan.summary_required,
    }


def _save_assistant_message(
    *,
    response: ProviderResponse,
    session: Session,
    run: Run,
    store: Store,
) -> Message:
    metadata = dict(response.raw)
    metadata["finish_reason"] = response.finish_reason
    if response.replay_state is not None:
        metadata["provider_replay_state"] = response.replay_state.to_metadata()

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
    if response.replay_state is not None:
        compaction_items = sum(
            item.get("type") == "compaction" for item in response.replay_state.items
        )
        if compaction_items:
            store.append_event(
                Event(
                    type="context.compacted",
                    session_id=session.id,
                    run_id=run.id,
                    message_id=message.id,
                    payload={
                        "provider": response.replay_state.provider,
                        "item_count": compaction_items,
                    },
                )
            )
    return message


def _provider_response_error(response: ProviderResponse) -> str | None:
    reason = response.finish_reason
    if response.tool_calls:
        if reason == "tool_use":
            return None
        return f"Provider returned tool calls with finish reason: {reason}"

    if reason == "stop":
        return None
    if reason == "tool_use":
        return "Provider returned tool_use without any tool calls"
    if reason == "max_tokens":
        return "Provider response stopped at a token limit"
    if reason == "refusal":
        return "Provider refused the request"
    if reason == "pause":
        return "Provider paused without a resumable tool call"
    return "Provider returned an unknown finish reason"


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


def _block_run(
    *,
    run: Run,
    final_message: Message,
    decisions: list[ToolCallDecision],
    iterations: int,
    store: Store,
) -> list[ApprovalRequest]:
    approvals = [
        ApprovalRequest(
            session_id=run.session_id,
            run_id=run.id,
            tool_call=item.tool_call,
            reason=item.decision.reason,
        )
        for item in decisions
        if item.decision.action == "ask"
    ]
    if not approvals:
        raise ValueError("Cannot block a run without pending approvals")

    run.status = "blocked"
    run.error = None
    run.updated_at = utc_now()

    events: list[Event] = []
    for approval in approvals:
        tool_call = approval.tool_call
        events.append(
            Event(
                type="approval.requested",
                session_id=run.session_id,
                run_id=run.id,
                message_id=final_message.id,
                payload={
                    "approval_id": approval.id,
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "reason": approval.reason,
                },
            )
        )

    events.append(
        Event(
            type="run.blocked",
            session_id=run.session_id,
            run_id=run.id,
            message_id=final_message.id,
            payload={
                "approval_ids": [approval.id for approval in approvals],
                "approval_count": len(approvals),
                "iterations": iterations,
            },
        )
    )
    store.block_run(run=run, approvals=approvals, events=events)
    return approvals


def _request_approvals(
    *,
    run: Run,
    final_message: Message,
    decisions: list[ToolCallDecision],
    iterations: int,
    store: Store,
    callback: ApprovalCallback | None,
) -> dict[str, ApprovalRequest]:
    approvals = _block_run(
        run=run,
        final_message=final_message,
        decisions=decisions,
        iterations=iterations,
        store=store,
    )
    resolved: list[ApprovalRequest] = []
    try:
        for approval in approvals:
            choice = callback(approval) if callback is not None else "denied"
            resolved.append(store.resolve_approval(approval.id, choice))
    except Exception:
        resolved_ids = {approval.id for approval in resolved}
        for approval in approvals:
            if approval.id not in resolved_ids:
                _ = store.resolve_approval(approval.id, "denied")
        raise

    run.status = "running"
    run.error = None
    run.updated_at = utc_now()
    store.resume_run(
        run=run,
        event=Event(
            type="run.resumed",
            session_id=run.session_id,
            run_id=run.id,
            message_id=final_message.id,
            payload={
                "approval_ids": [approval.id for approval in resolved],
                "decisions": [approval.status for approval in resolved],
                "iterations": iterations,
            },
        ),
    )

    return {approval.tool_call.id: approval for approval in resolved}


def _deny_tool_call(
    *,
    decision: ToolCallDecision,
    session: Session,
    run: Run,
    store: Store,
    approval: ApprovalRequest | None = None,
    denial_reason: str | None = None,
) -> Message:
    content = decision.decision.reason or "Tool call denied by policy"
    metadata: dict[str, object] = {
        "ok": False,
        "policy_action": "deny",
    }
    if approval is not None:
        content = denial_reason or "Tool call denied by user"
        metadata["approval_id"] = approval.id
        metadata["approval_status"] = approval.status

    message = Message(
        session_id=session.id,
        run_id=run.id,
        role="tool",
        content=content,
        name=decision.tool_call.name,
        tool_call_id=decision.tool_call.id,
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
                "name": message.name,
                "ok": False,
                "policy_action": "deny",
            },
        )
    )
    return message


def _execute_tool_call(
    *,
    tool_call: ToolCall,
    session: Session,
    run: Run,
    registry: ToolRegistry,
    context: ExecutionContext,
    store: Store,
) -> Message:
    _record_tool_started(
        tool_call=tool_call,
        session=session,
        run=run,
        store=store,
    )
    result = registry.execute(tool_call, context)
    return _save_tool_result(
        result=result,
        session=session,
        run=run,
        store=store,
    )


def _record_tool_started(
    *,
    tool_call: ToolCall,
    session: Session,
    run: Run,
    store: Store,
) -> None:
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


def _save_tool_result(
    *,
    result: ToolResult,
    session: Session,
    run: Run,
    store: Store,
) -> Message:
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


def _execute_parallel_tool_calls(
    *,
    tool_calls: list[ToolCall],
    session: Session,
    run: Run,
    registry: ToolRegistry,
    context: ExecutionContext,
    store: Store,
) -> list[Message]:
    if len(tool_calls) < 2:
        return [
            _execute_tool_call(
                tool_call=tool_calls[0],
                session=session,
                run=run,
                registry=registry,
                context=context,
                store=store,
            )
        ]

    max_workers = min(len(tool_calls), MAX_PARALLEL_TOOL_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future[ToolResult]] = []
        for tool_call in tool_calls:
            _record_tool_started(
                tool_call=tool_call,
                session=session,
                run=run,
                store=store,
            )
            futures.append(executor.submit(registry.execute, tool_call, context))

        results = [future.result() for future in futures]

    return [
        _save_tool_result(
            result=result,
            session=session,
            run=run,
            store=store,
        )
        for result in results
    ]


def _execute_tool_decisions(
    *,
    decisions: list[ToolCallDecision],
    approvals_by_tool_call: dict[str, ApprovalRequest],
    session: Session,
    run: Run,
    registry: ToolRegistry,
    context: ExecutionContext,
    store: Store,
    approval_callback: ApprovalCallback | None,
) -> list[Message]:
    messages: list[Message] = []
    parallel_calls: list[ToolCall] = []

    def flush_parallel_calls() -> None:
        if not parallel_calls:
            return
        messages.extend(
            _execute_parallel_tool_calls(
                tool_calls=parallel_calls,
                session=session,
                run=run,
                registry=registry,
                context=context,
                store=store,
            )
        )
        parallel_calls.clear()

    for decision in decisions:
        approval = approvals_by_tool_call.get(decision.tool_call.id)
        denied = decision.decision.action == "deny" or (
            approval is not None and approval.status == "denied"
        )
        allowed = decision.decision.action == "allow" or (
            approval is not None and approval.status == "approved"
        )

        if denied:
            flush_parallel_calls()
            messages.append(
                _deny_tool_call(
                    decision=decision,
                    session=session,
                    run=run,
                    store=store,
                    approval=approval,
                    denial_reason=(
                        "Tool call denied because no approval callback is configured"
                        if approval is not None and approval_callback is None
                        else None
                    ),
                )
            )
            continue

        if not allowed:
            raise RuntimeError("Unresolved approval reached tool execution")

        registered_tool = registry.get(decision.tool_call.name)
        if registered_tool is not None and registered_tool.parallel_safe:
            parallel_calls.append(decision.tool_call)
            continue

        flush_parallel_calls()
        messages.append(
            _execute_tool_call(
                tool_call=decision.tool_call,
                session=session,
                run=run,
                registry=registry,
                context=context,
                store=store,
            )
        )

    flush_parallel_calls()
    return messages


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
    policy: ApprovalPolicy | None = None,
    approval_callback: ApprovalCallback | None = None,
    parent_run_id: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> RunOutcome:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    allowed_registry = registry.subset(agent.tools)
    messages = store.list_messages(session.id)
    if not messages:
        raise ValueError("Cannot run an agent without session messages")

    approval_policy = policy if policy is not None else DefaultApprovalPolicy()
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
            request = _plan_provider_request(
                agent=agent,
                session=session,
                run=run,
                messages=messages,
                registry=allowed_registry,
                provider=provider,
                store=store,
                iteration=iterations,
            )
            response = provider.generate(request)

            final_message = _save_assistant_message(
                response=response,
                session=session,
                run=run,
                store=store,
            )
            messages.append(final_message)

            response_error = _provider_response_error(response)
            if response_error is not None:
                return _fail_run(
                    run=run,
                    error=response_error,
                    iterations=iterations,
                    store=store,
                    final_message=final_message,
                )

            if not final_message.tool_calls:
                return _finish_run(
                    run=run,
                    final_message=final_message,
                    iterations=iterations,
                    store=store,
                )

            decisions = _preflight_tool_calls(
                tool_calls=final_message.tool_calls,
                registry=allowed_registry,
                context=context,
                policy=approval_policy,
            )
            approvals_by_tool_call: dict[str, ApprovalRequest] = {}
            if any(item.decision.action == "ask" for item in decisions):
                approvals_by_tool_call = _request_approvals(
                    run=run,
                    final_message=final_message,
                    decisions=decisions,
                    iterations=iterations,
                    store=store,
                    callback=approval_callback,
                )

            messages.extend(
                _execute_tool_decisions(
                    decisions=decisions,
                    approvals_by_tool_call=approvals_by_tool_call,
                    session=session,
                    run=run,
                    registry=allowed_registry,
                    context=context,
                    store=store,
                    approval_callback=approval_callback,
                )
            )

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
