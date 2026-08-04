# Approval Enforcement PRD

## Summary

Approval enforcement is the policy boundary between a model requesting a tool and Conveyor allowing that tool to create side effects. Read-only tools may execute automatically. Write and dangerous tools must block the run, persist an approval request, and wait for an explicit decision before execution.

Approvals are durable runtime state, not an in-process confirmation callback. A desktop client, gateway, or future automation can resolve the request later, and Conveyor can resume the same run without asking the model to recreate its tool call.

## Goals

- Prevent tools that require approval from executing before an approval is persisted and granted.
- Keep blocked runs inspectable and resumable across process restarts.
- Preserve the exact tool name, arguments, call ID, session, and run that were approved.
- Return denied tool calls to the model as tool results so it can explain, recover, or choose another approach.
- Emit enough durable events for a desktop approval UI and runtime timeline.
- Keep approval policy separate from tool implementation and provider adapters.

## Non-Goals

- Building write, edit, or shell tools.
- User authentication or authorization for remote approval clients.
- Remembering decisions such as "always allow this tool."
- Path-specific, command-specific, or environment-specific policy rules.
- Parallel tool execution.
- Guaranteeing exactly-once side effects across arbitrary process crashes.

## Initial Policy

Tool permissions keep their current meanings:

- `read`: execute automatically.
- `write`: require approval.
- `dangerous`: require approval.

The initial policy is fixed and local. It should be represented behind a small policy interface so profiles and environment-specific rules can be added later without changing the agent loop.

The model does not decide whether approval is needed. Tool permission is trusted runtime metadata attached by the tool author.

## Lifecycle

```text
assistant Message with ToolCall[]
  -> persist assistant message
  -> preflight the complete tool batch
      -> all calls are read-only
          -> execute batch
          -> persist tool messages
          -> continue model loop
      -> one or more calls require approval
          -> persist ApprovalRequest for each risky call
          -> mark Run blocked
          -> emit approval.requested and run.blocked
          -> return blocked RunOutcome

approval decisions arrive later
  -> resolve each pending ApprovalRequest
  -> emit approval.resolved
  -> resume the same Run
      -> approved call: execute original ToolCall
      -> denied call: persist failed tool result without execution
      -> preserve tool-result order from the assistant message
      -> continue model loop
```

## State Transitions

Run transitions:

```text
pending -> running -> finished
                   -> failed
                   -> blocked -> running
                              -> cancelled
```

Approval transitions:

```text
pending -> approved
        -> denied
```

Approval decisions are immutable. Resolving an already resolved approval to a different status must fail. Repeating the same resolution may return the existing approval as an idempotent operation.

## Batch Semantics

The complete assistant tool-call batch must be preflighted before any call in that batch executes. If one call requires approval, Conveyor blocks before executing read-only calls from the same batch.

This rule avoids partially executed batches and makes resume deterministic. Once all approvals in the batch are resolved:

- approved and read-only calls execute sequentially
- denied calls become failed tool-result messages
- tool-result messages are persisted in the assistant's original call order
- the model is called again only after every call has a corresponding result

## Denial Semantics

A denied call is not a failed run. Conveyor creates a tool message correlated to the original `tool_call_id`:

```text
role: tool
name: original tool name
content: Tool call denied by user
metadata:
  ok: false
  approval_id: approval ID
  approval_status: denied
```

The model receives this result and may continue without the action, request a safer alternative, or explain that it cannot complete the task.

## Crash Safety

Conveyor must persist in this order:

1. assistant message containing the tool call
2. approval request
3. blocked run state
4. approval decision
5. `tool.started` event
6. tool result message
7. `tool.finished` event

A risky tool must never execute if steps 1 through 4 are not durable.

If Conveyor restarts after `tool.started` but before a tool result is persisted, it must not automatically execute that side-effecting call again. The run should surface an uncertain execution state for manual reconciliation. Exactly-once execution requires idempotent tools or backend-specific transaction support and is deferred.

## Model Changes

`agent/models.py`:

- Add an `ApprovalStatus` alias.
- Add `run.blocked`, `run.resumed`, and `approval.resolved` event types.
- Keep `ApprovalRequest` bound to one exact `ToolCall` and one run.
- Extend `RunOutcome` with `pending_approvals: list[ApprovalRequest]` so clients do not need an immediate follow-up query.

No provider-specific approval types should enter the agent models.

## Policy Layer

Add `agent/approvals.py` with a small policy and lifecycle boundary:

```text
ApprovalPolicy
  -> requires_approval(tool permission) -> bool

request_approvals(...)
  -> creates durable requests
  -> blocks the run

resolve_approval(...)
  -> performs an atomic pending-to-approved/denied transition
  -> emits approval.resolved
```

The policy layer decides whether execution is allowed. `ToolRegistry` continues to own tool lookup and invocation.

## Storage Changes

`storage/store.py` needs:

- `list_approvals(run_id=None, status=None)`
- `resolve_approval(approval_id, decision)` with an atomic conditional update
- a way to retrieve the latest assistant tool-call batch for a blocked run
- tests for pending, approved, denied, repeated, and conflicting resolutions

The existing approvals table is sufficient for the initial implementation. No migration is required while the project has no released database schema. Schema versioning must be revisited before persistent user databases exist.

## Loop Changes

`agent/loop.py` needs:

- `_preflight_tool_calls()` to inspect the full batch before execution
- `_block_run()` to persist approvals, update the run, emit events, and return a blocked outcome
- `_tool_result_for_denial()` to produce a model-visible denial
- `_execute_tool_batch()` to preserve call order after decisions
- `resume_agent_run()` to continue a blocked run from its persisted assistant message and approval decisions

`run_agent()` remains responsible for starting a new run. Resume must not create a replacement run or call the provider before resolving the blocked tool batch.

## Store and Event Invariants

- Every approval references an existing session and run.
- Every approval stores the original tool call unchanged.
- An approval decision cannot modify tool arguments.
- A blocked run has at least one pending approval.
- A run cannot resume while any approval in its current batch is pending.
- A risky tool cannot produce `tool.started` without an approved request.
- A denied tool cannot produce `tool.started`.
- Every approved, denied, or read-only call receives exactly one tool-result message in normal execution.
- Events are appended after the state they describe has been persisted.

## Client Contract

The runtime-facing API should eventually expose:

```text
list pending approvals
get approval details
approve approval ID
deny approval ID
resume blocked run
subscribe to approval and run events
```

The interactive smoke runner can temporarily provide `approve` and `deny` prompts, but approval decisions must still pass through the same durable policy functions used by future clients.

## Tests

Required loop tests:

- read tool executes without approval
- write tool blocks before execution
- dangerous tool blocks before execution
- mixed read/write batch executes nothing before approval
- approved tool executes once and the run continues
- denied tool does not execute and the model receives a denial result
- multiple risky calls wait until every decision is resolved
- provider is not called again while the run is blocked
- unresolved approval survives store reload
- conflicting approval resolution fails
- provider failure after resume marks the same run failed
- iteration accounting remains correct across block and resume

## Implementation Plan

1. Extend approval, event, and outcome models.
2. Add storage listing and atomic resolution methods with tests.
3. Add the fixed initial `ApprovalPolicy`.
4. Add batch preflight and blocked-run helpers.
5. Make `run_agent()` return a blocked outcome before risky execution.
6. Add denial tool-result construction.
7. Add `resume_agent_run()` using the persisted assistant tool-call batch.
8. Test mixed batches, approval, denial, and restart behavior with fake tools and `FakeProvider`.
9. Extend the interactive smoke runner to resolve approvals.
10. Add write and shell tools only after these acceptance criteria pass.

## Acceptance Criteria

- No `write` or `dangerous` tool executes without a durable approved request.
- Blocking and approval decisions remain available after reopening SQLite storage.
- Approval and denial both resume the original run deterministically.
- Denial is visible to the model and does not fail the run by itself.
- Mixed tool batches cannot partially execute before approval.
- The provider is not called while a run is waiting for approval.
- Events provide enough information for a client to render pending and resolved approvals.
- All existing read-tool and agent-loop tests continue to pass.
