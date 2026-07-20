# Conveyor

Conveyor is a self-hosted agent environment for operating always-on assistants.

It is designed around long-lived agents that can be reached through desktop
apps, chat gateways, webhooks, and other clients while keeping durable state,
tool history, approvals, and runtime events under the user's control.

Conveyor treats the agent as an operated product, not just a library call. The
runtime owns sessions, model interaction, tool execution, policy, event history,
and the context needed for agents to improve over time through memory, skills,
feedback, and replay.

## Definition

Conveyor is:

- a self-hosted runtime for persistent agents
- a gateway environment for talking to agents from multiple surfaces
- a policy-controlled tool execution layer
- a durable event and transcript system
- a foundation for memory, skills, automations, and multi-agent workflows
- a local-first environment for self-improving assistants

## Principles

- The runtime is the product; interfaces are clients.
- Durable events are the source of truth.
- Prompt context is derived from stored history.
- Tool use should be observable and permissioned.
- Agent capabilities should grow as modular subsystems.

