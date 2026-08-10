# approval enforcement PRD

## summary

Approval enforcement is the policy boundary between a model requesting a tool
and Conveyor allowing that tool to create side effects. Read-only tools may run
automatically. Write and dangerous tools require a decision from the active
client before execution.

Approval waits are part of the live agent run. The current `run_agent()` call
remains active while a CLI, desktop client, or gateway callback obtains the
user's decision. Conveyor persists approval state and events for UI rendering
and auditing, but does not reconstruct an interrupted Python call stack after a
daemon restart.

## goals

- Prevent risky tools from executing without explicit approval.
- Keep policy separate from tools and provider adapters.
- Persist approval requests, decisions, and run-state events for observability.
- Return denied calls to the model as correlated tool results.
- Preserve the model's original tool-call order.
- Use one callback contract across CLI, desktop, and gateway clients.

## non-goals

- Resuming an approval wait after the daemon exits.
- Reconstructing a tool batch from stored events.
- Guaranteeing exactly-once side effects across process crashes.
- Remembering decisions such as "always allow."
- User authentication for remote approval clients.
- Parallel tool execution.

## policy

- `read`: allow automatically.
- `write`: ask the active user.
- `dangerous`: ask the active user.
- Unknown tools: deny automatically.

The model cannot choose its own permission. Permission is trusted runtime
metadata attached by the tool author.

## lifecycle

```text
assistant tool-call batch
  -> persist assistant message
  -> preflight every call
  -> no call requires approval
       -> execute allowed calls and return denied results
  -> one or more calls require approval
       -> persist ApprovalRequest rows
       -> mark Run blocked
       -> emit approval.requested and run.blocked
       -> invoke the approval callback for each request
       -> persist approved or denied decisions
       -> mark Run running
       -> emit approval.resolved and run.resumed
       -> execute approved calls and return denied results
  -> append all tool messages in original order
  -> continue the same provider loop
```

`run_agent()` does not return while approval is pending. The callback may block
on terminal input, a `threading.Event`, or another live synchronization
primitive. If no callback is configured, Conveyor denies the request rather
than executing without consent.

## state transitions

```text
Run:      running -> blocked -> running -> finished|failed
Approval: pending -> approved|denied
```

Approval decisions are immutable. Repeating the same decision is idempotent;
attempting to replace it with the opposite decision fails.

## batch semantics

Conveyor preflights the complete assistant batch before executing any call. If
one call needs approval, the callback resolves every required approval before
the batch begins. Conveyor then processes calls sequentially in their original
order:

- `allow`: execute.
- `deny`: create a failed tool result without execution.
- `ask` plus `approved`: execute.
- `ask` plus `denied`: create a failed tool result without execution.

The model is called again only after every requested call has a corresponding
tool message.

## callback contract

```python
class ApprovalCallback(Protocol):
    def __call__(self, approval: ApprovalRequest, /) -> ApprovalDecision: ...
```

The core callback is synchronous because the current agent loop is synchronous.
A desktop or gateway adapter can bridge asynchronous user input by notifying
the UI and waiting on a thread-safe primitive. The agent run should execute on
a worker thread so one approval does not block the daemon's event loop.

## persistence and events

Conveyor stores:

- The exact tool-call ID, name, and arguments in `ApprovalRequest`.
- `approval.requested` when the request becomes visible.
- `run.blocked` before the callback begins waiting.
- `approval.resolved` for the callback result.
- `run.resumed` before tool execution continues.
- Normal `tool.started`, tool message, and `tool.finished` records afterward.

Persistence provides an audit trail and real-time UI state. It is not a durable
continuation mechanism in this phase. On startup, a later recovery task should
mark leftover `running` or `blocked` runs as interrupted or failed.

## acceptance criteria

- Read tools execute without invoking the approval callback.
- Write and dangerous tools cannot execute before approval.
- The run is visibly blocked while the callback is active.
- Approved calls execute once in the live run.
- Denied calls never emit `tool.started`.
- A denial is returned to the model and does not fail the run by itself.
- Missing approval callbacks fail closed.
- Mixed batches execute in their original order after approvals resolve.
- Approval and run-state events provide enough information for the desktop UI.
- Existing provider and read-tool behavior remains unchanged.
