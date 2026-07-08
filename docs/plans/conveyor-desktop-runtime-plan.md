# Conveyor Project Definition

## Product Thesis

Conveyor is a self-hosted, always-on agent runtime: an agent as a product you operate, not just a library you build with.

It should run locally or on infrastructure the user controls, keep long-lived sessions, execute tools under policy, expose gateways for humans and apps to talk to it, and preserve enough history for inspection, replay, memory, and future self-improvement.

The desktop app is the first client, not the whole product. Conveyor should also be able to grow into chat-app gateways, automations, CLI/admin tooling, and multi-agent workflows without changing the core identity of the project.

## Core Capabilities

- Always-on runtime with health, configuration, profiles, and durable state.
- Conversational sessions that can continue over time instead of one-off prompt calls.
- Agent runs that use models, tools, events, approvals, and persistent transcripts.
- Tool execution under explicit policy, especially for file writes, shell commands, credentials, and external integrations.
- Gateway layer for clients and platforms to send messages, receive events, and resolve approvals.
- Event-first observability so every meaningful model action, tool call, failure, and approval can be inspected later.
- Context management that treats the transcript as durable history and prompt context as a derived, bounded view.

## Product Shape

```text
Conveyor runtime
  -> agent core
  -> tools
  -> model providers
  -> storage
  -> gateway interfaces

Desktop app
  -> first control surface
  -> chat, event timeline, approvals, settings

Future clients
  -> CLI/admin tools
  -> web UI
  -> chat apps
  -> webhooks
  -> scheduled automations
```

## Design Principles

- Runtime first: clients are replaceable surfaces over the same operated agent.
- Durable state over prompt history: store complete events, derive compact context per run.
- Observable execution: tool calls, approvals, retries, and failures should be visible.
- Policy before power: dangerous actions must pass through permissions and approvals.
- Modular integrations: model providers, tools, gateways, memory, and skills should be replaceable subsystems.
- Start single-agent, stay multi-agent compatible: the first runtime can be simple without blocking delegation later.
- Avoid premature platform sprawl: build one gateway and one client well before adding many adapters.

## First Milestone

The first working version should prove the core loop:

```text
desktop client
  -> sends a message to the local runtime
  -> runtime creates or continues a session
  -> agent calls a model or fake provider
  -> agent may call a tool
  -> runtime emits visible events
  -> risky actions can request approval
  -> final answer returns to the client
  -> session and events persist
```

This milestone should be daemon-first with the desktop app as the first user-facing client. CLI can come later as an admin/debug surface.

## Initial Defaults

- Python runtime managed with `uv`.
- Electron + React for the first desktop client.
- FastAPI/WebSocket for the first local gateway.
- SQLite or another local embedded store for early durable state.
- One configurable OpenAI-compatible provider plus a fake provider for tests.
- Initial tools should stay minimal: inspect files, search files, write files under policy, and run commands under policy.
- Memory, skills, cron, external chat gateways, MCP, remote environments, and true multi-agent orchestration are later features.
