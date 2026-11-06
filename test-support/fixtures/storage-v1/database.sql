BEGIN TRANSACTION;
CREATE TABLE approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    run_id TEXT NOT NULL REFERENCES runs(id),
    tool_call TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'denied')),
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
INSERT INTO "approvals" VALUES('appr_fixture','ses_fixture','run_b','{"id": "provider_write_id", "name": "write_file", "arguments": {"path": "notes/\u4f60\u597d.md", "content": "caf\u00e9 \ud83e\udd80\n"}}','write access requires approval','pending','2026-10-26T17:30:00.123456+05:30',NULL);
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    session_id TEXT REFERENCES sessions(id),
    run_id TEXT REFERENCES runs(id),
    message_id TEXT REFERENCES messages(id),
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
INSERT INTO "events" VALUES('evt_a','session.created','ses_fixture',NULL,NULL,'{}','2026-10-26T12:00:00.123456+00:00');
INSERT INTO "events" VALUES('evt_b','run.blocked','ses_fixture','run_b',NULL,'{"approval_ids": ["appr_fixture"]}','2026-10-26T17:30:00.123456+05:30');
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id),
    name TEXT,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_call_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
INSERT INTO "messages" VALUES('msg_a','ses_fixture','user','Inspect the workspace',NULL,NULL,'[]',NULL,'{}','2026-10-26T12:00:00.123456+00:00');
INSERT INTO "messages" VALUES('msg_b','ses_fixture','assistant','Reading','run_a',NULL,'[{"id": "provider_read_id", "name": "read_file", "arguments": {"path": "notes/\u4f60\u597d.md"}}]',NULL,'{"provider_replay_state": {"provider": "anthropic", "items": [{"type": "compaction", "content": null, "encrypted_content": "opaque-fixture"}]}, "nested": {"array": [true, null, 2.5]}}','2026-10-26T12:00:00.123456+00:00');
INSERT INTO "messages" VALUES('msg_c','ses_fixture','tool','café 🦀','run_a','read_file','[]','provider_read_id','{"ok": true, "permission": "read"}','2026-10-26T12:00:00.123456+00:00');
INSERT INTO "messages" VALUES('msg_d','ses_fixture','assistant','Writing','run_b',NULL,'[{"id": "provider_write_id", "name": "write_file", "arguments": {"path": "notes/\u4f60\u597d.md", "content": "caf\u00e9 \ud83e\udd80\n"}}]',NULL,'{}','2026-10-26T17:30:00.123456+05:30');
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO "meta" VALUES('schema_version','1');
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    agent_id TEXT NOT NULL,
    parent_run_id TEXT REFERENCES runs(id),
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO "runs" VALUES('run_a','ses_fixture','agent_fixture',NULL,'finished',NULL,'2026-10-26T12:00:00.123456+00:00','2026-10-26T12:00:00.123456+00:00');
INSERT INTO "runs" VALUES('run_b','ses_fixture','agent_fixture','run_a','blocked',NULL,'2026-10-26T17:30:00.123456+05:30','2026-10-26T17:30:00.123456+05:30');
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_source TEXT NOT NULL DEFAULT 'default'
        CHECK (title_source IN ('default', 'auto', 'user')),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO "sessions" VALUES('ses_fixture','Migration café 🦀','default','active','2026-10-26T12:00:00.123456+00:00','2026-10-26T17:30:00.123456+05:30');
CREATE INDEX idx_runs_session ON runs(session_id);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);
CREATE INDEX idx_events_session ON events(session_id, created_at);
CREATE INDEX idx_events_run ON events(run_id, created_at);
CREATE INDEX idx_approvals_run ON approvals(run_id);
COMMIT;
