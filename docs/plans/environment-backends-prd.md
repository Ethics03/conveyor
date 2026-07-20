# Environment Backends PRD

## Summary

Environment backends define where Conveyor tools execute. The local filesystem and local shell are useful for early development, but they are not a security boundary. Conveyor should eventually route file, search, write, and command tools through explicit execution environments.

The first implementation can stay local. This PRD defines the future shape so current tools do not accidentally assume host-local execution forever.

## Product Requirements

- Conveyor can run tools against different execution environments without changing the agent loop.
- The local environment remains the default for development and trusted personal use.
- Sandboxed or remote environments can be added for untrusted tasks, CI-style workflows, remote machines, or disposable workspaces.
- Tool calls should expose the same high-level behavior regardless of backend: read files, search files, write files, run commands, and return structured results.
- Environment backends own filesystem visibility, process execution, working directory, environment variables, resource limits, and teardown behavior.
- The agent loop should not know whether a tool ran locally, in a container, or remotely.

## Architecture

```text
agent loop
  -> ToolRegistry
  -> workspace/command tool
  -> EnvironmentBackend
      -> local backend
      -> container backend later
      -> remote backend later
  -> structured tool result
```

Current local helpers such as path resolution are guardrails. They should not be treated as the final security model.

## Backend Interface

Future backends should converge around a small capability interface:

```text
read_file(path, offset, limit)
search_files(pattern, target, path, file_glob, limit, offset, context)
write_file(path, content)
run_command(command, cwd, timeout, env)
```

Backends may implement these directly or translate them into shell commands inside the environment. Tool functions should call the backend interface rather than reaching into host-specific filesystem/process APIs.

## Initial Backend Types

- Local backend
  - Uses host filesystem and subprocesses.
  - Intended for trusted development and personal use.
  - Uses workspace path containment as a guardrail.

- Container backend
  - Mounts the selected workspace into a container.
  - Runs file/search/command operations inside that container.
  - Supports resource limits, network policy, and disposable or persistent workspace state.

- Remote backend
  - Executes tools on a remote machine or managed workspace.
  - Keeps the same tool protocol while changing where execution happens.
  - May require sync or artifact export for files changed remotely.

## Design Principles

- The environment is the security boundary, not string validation inside the agent process.
- Tool APIs should be stable even when execution moves from local to sandboxed or remote backends.
- Secrets and host credentials should not be forwarded by default.
- Backend configuration should be explicit and profile-scoped.
- Local execution should stay simple, but not leak assumptions into the agent loop.
- Tool results should include enough metadata to inspect where and how execution happened.

## Deferred Work

- Backend configuration schema.
- Container image selection and lifecycle management.
- Resource limits for CPU, memory, disk, network, and process count.
- Persistent vs ephemeral environment policy.
- Remote file sync or artifact export.
- Approval policy differences by backend.
- UI controls for choosing and inspecting the active environment.
