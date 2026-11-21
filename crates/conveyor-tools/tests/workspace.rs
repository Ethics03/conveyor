use std::{
    fs,
    path::{Path, PathBuf},
};

use conveyor_core::{
    CancellationToken,
    models::{JsonObject, ToolCall},
    tools::{DynTool, ToolContext, ToolOutput},
};
use conveyor_tools::{ReadFile, ReadMany, SearchFiles, default_registry};
use serde_json::{Value, json};
use tempfile::TempDir;

/// A workspace with a sibling directory outside it.
struct Fixture {
    _dir: TempDir,
    workspace: PathBuf,
    outside: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let workspace = dir.path().join("workspace");
        let outside = dir.path().join("outside");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret\n").unwrap();
        Self {
            _dir: dir,
            workspace,
            outside,
        }
    }

    fn write(&self, path: &str, content: impl AsRef<[u8]>) {
        let path = self.workspace.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    async fn call(&self, tool: &dyn DynTool, arguments: Value) -> Result<Value, String> {
        let Value::Object(arguments) = arguments else {
            panic!("arguments must be an object");
        };
        let context = ToolContext::new(&self.workspace, CancellationToken::new());
        match tool.execute(arguments, &context).await {
            Ok(ToolOutput::Json(value)) => Ok(value),
            Ok(other) => panic!("expected JSON output, got {other:?}"),
            Err(error) => Err(error.to_string()),
        }
    }
}

fn ripgrep_installed() -> bool {
    std::process::Command::new("rg")
        .arg("--version")
        .output()
        .is_ok()
}

#[tokio::test]
async fn read_file_returns_numbered_lines_with_metadata() {
    let fixture = Fixture::new();
    fixture.write("notes.txt", "alpha\nbeta\ngamma\n");

    let result = fixture
        .call(&ReadFile, json!({"path": "notes.txt"}))
        .await
        .unwrap();

    assert_eq!(
        result,
        json!({
            "path": "notes.txt",
            "content": "1|alpha\n2|beta\n3|gamma",
            "offset": 1,
            "limit": 500,
            "total_lines": 3,
            "total_lines_is_exact": true,
            "truncated": false,
        })
    );
}

#[tokio::test]
async fn read_file_pages_and_looks_one_line_ahead() {
    let fixture = Fixture::new();
    fixture.write("notes.txt", "one\ntwo\nthree\nfour\nfive\n");

    let page = fixture
        .call(
            &ReadFile,
            json!({"path": "notes.txt", "offset": 2, "limit": 2}),
        )
        .await
        .unwrap();
    assert_eq!(page["content"], "2|two\n3|three");
    assert_eq!(page["offset"], 2);
    assert_eq!(page["limit"], 2);
    assert_eq!(page["total_lines"], 4);
    assert_eq!(page["total_lines_is_exact"], false);
    assert_eq!(page["next_offset"], 4);
    assert_eq!(page["truncated"], true);
    assert_eq!(page["hint"], "Continue with offset=4.");

    let last = fixture
        .call(
            &ReadFile,
            json!({"path": "notes.txt", "offset": 4, "limit": 9}),
        )
        .await
        .unwrap();
    assert_eq!(last["content"], "4|four\n5|five");
    assert_eq!(last["total_lines"], 5);
    assert_eq!(last["total_lines_is_exact"], true);
    assert_eq!(last["truncated"], false);
    assert!(last.get("next_offset").is_none());

    // Out-of-range paging arguments are clamped.
    let clamped = fixture
        .call(
            &ReadFile,
            json!({"path": "notes.txt", "offset": -3, "limit": 0}),
        )
        .await
        .unwrap();
    assert_eq!(clamped["content"], "1|one");
    assert_eq!(clamped["offset"], 1);
    assert_eq!(clamped["limit"], 1);
}

#[tokio::test]
async fn read_file_clips_long_lines_and_bounds_output() {
    let fixture = Fixture::new();
    fixture.write("minified.txt", format!("{}\nsecond\n", "x".repeat(100_100)));

    let result = fixture
        .call(&ReadFile, json!({"path": "minified.txt"}))
        .await
        .unwrap();

    let content = result["content"].as_str().unwrap();
    let first = content.lines().next().unwrap();
    assert!(first.starts_with("1|"));
    assert!(first.ends_with(" ... [truncated]"));
    assert_eq!(
        first.chars().count(),
        "1|".len() + 2_000 + " ... [truncated]".len()
    );
    assert_eq!(result["total_lines"], 2);
    assert_eq!(result["total_lines_is_exact"], true);
    assert_eq!(result["line_truncated"], true);
    assert_eq!(result["truncated"], true);
    assert_eq!(
        result["hint"],
        "Lines longer than 2,000 characters were clipped."
    );
}

#[tokio::test]
async fn read_file_handles_crlf_unicode_and_invalid_text() {
    let fixture = Fixture::new();
    fixture.write("windows.txt", "café\r\n🦀\r\n");
    fixture.write("binary.bin", [b'o', b'k', b'\n', 0xff, 0xfe]);
    fixture.write("dir/inner.txt", "inner\n");

    let windows = fixture
        .call(&ReadFile, json!({"path": "windows.txt"}))
        .await
        .unwrap();
    assert_eq!(windows["content"], "1|café\n2|🦀");

    let error = fixture
        .call(&ReadFile, json!({"path": "binary.bin"}))
        .await
        .unwrap_err();
    assert_eq!(error, "binary.bin is not valid UTF-8 text");

    let error = fixture
        .call(&ReadFile, json!({"path": "dir"}))
        .await
        .unwrap_err();
    assert_eq!(error, "Not a file: dir");
}

#[tokio::test]
async fn read_file_rejects_offsets_past_the_end_and_unknown_arguments() {
    let fixture = Fixture::new();
    fixture.write("notes.txt", "one\ntwo\n");

    let error = fixture
        .call(&ReadFile, json!({"path": "notes.txt", "offset": 3}))
        .await
        .unwrap_err();
    assert_eq!(error, "Offset 3 exceeds the file's 2 lines");

    let error = fixture
        .call(&ReadFile, json!({"path": "notes.txt", "lines": 3}))
        .await
        .unwrap_err();
    assert!(error.starts_with("Invalid arguments for read_file: unknown field `lines`"));
}

#[tokio::test]
async fn paths_resolve_inside_the_workspace_only() {
    let fixture = Fixture::new();
    fixture.write("src/notes.txt", "inside\n");
    let absolute_inside = fixture.workspace.join("src/notes.txt");
    let absolute_outside = fixture.outside.join("secret.txt");

    for path in [
        "src/notes.txt",
        "./src/../src/notes.txt",
        absolute_inside.to_str().unwrap(),
    ] {
        let result = fixture
            .call(&ReadFile, json!({"path": path}))
            .await
            .unwrap();
        assert_eq!(result["path"], "src/notes.txt", "{path}");
    }
    for path in [
        "../outside/secret.txt",
        "src/../../outside/secret.txt",
        absolute_outside.to_str().unwrap(),
    ] {
        let error = fixture
            .call(&ReadFile, json!({"path": path}))
            .await
            .unwrap_err();
        assert_eq!(error, format!("Path escapes workspace: {path}"));
    }
}

#[cfg(unix)]
#[tokio::test]
async fn symbolic_links_cannot_lead_outside() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    let workspace = &fixture.workspace;
    fixture.write("real.txt", "real\n");
    symlink(
        fixture.outside.join("secret.txt"),
        workspace.join("file_link"),
    )
    .unwrap();
    symlink(&fixture.outside, workspace.join("dir_link")).unwrap();
    // Dangling: the target does not exist yet, so it could be created through.
    symlink(
        fixture.outside.join("later.txt"),
        workspace.join("dangling"),
    )
    .unwrap();
    symlink("real.txt", workspace.join("inside_link")).unwrap();

    for path in ["file_link", "dir_link/secret.txt", "dangling"] {
        let error = fixture
            .call(&ReadFile, json!({"path": path}))
            .await
            .unwrap_err();
        assert_eq!(error, format!("Path escapes workspace: {path}"));
    }
    let error = fixture
        .call(&SearchFiles, json!({"pattern": "*", "path": "dir_link"}))
        .await
        .unwrap_err();
    assert_eq!(error, "Path escapes workspace: dir_link");

    let inside = fixture
        .call(&ReadFile, json!({"path": "inside_link"}))
        .await
        .unwrap();
    assert_eq!(inside["path"], "real.txt");
    assert_eq!(inside["content"], "1|real");
}

#[tokio::test]
async fn read_many_returns_multiple_files() {
    let fixture = Fixture::new();
    fixture.write("a.txt", "alpha\n");
    fixture.write("b.txt", "beta\n");

    let result = fixture
        .call(&ReadMany, json!({"paths": ["a.txt", "b.txt"]}))
        .await
        .unwrap();

    let page = |path: &str, content: &str| {
        json!({
            "path": path,
            "content": content,
            "offset": 1,
            "limit": 500,
            "total_lines": 1,
            "total_lines_is_exact": true,
            "truncated": false,
        })
    };
    assert_eq!(
        result,
        json!({
            "files": [page("a.txt", "1|alpha"), page("b.txt", "1|beta")],
            "errors": [],
            "requested_count": 2,
            "processed_count": 2,
            "total_chars": 13,
            "max_total_chars": 100_000,
            "truncated": false,
        })
    );
}

#[tokio::test]
async fn read_many_reports_failed_files_beside_the_others() {
    let fixture = Fixture::new();
    fixture.write("present.txt", "hello\n");
    fixture.write("short.txt", "one\n");

    let result = fixture
        .call(
            &ReadMany,
            json!({"paths": ["missing.txt", "../outside/secret.txt", "present.txt"]}),
        )
        .await
        .unwrap();
    assert_eq!(result["files"][0]["path"], "present.txt");
    assert_eq!(
        result["errors"],
        json!([
            {"path": "missing.txt", "error": "Not a file: missing.txt"},
            {
                "path": "../outside/secret.txt",
                "error": "Path escapes workspace: ../outside/secret.txt",
            },
        ])
    );
    assert_eq!(result["processed_count"], 3);
    assert_eq!(result["truncated"], false);

    // An offset past one file's end fails only that file.
    let result = fixture
        .call(
            &ReadMany,
            json!({"paths": ["present.txt", "short.txt"], "offset": 2}),
        )
        .await
        .unwrap();
    assert_eq!(result["files"], json!([]));
    assert_eq!(
        result["errors"][1],
        json!({"path": "short.txt", "error": "Offset 2 exceeds the file's 1 lines"})
    );
}

#[tokio::test]
async fn read_many_enforces_its_budgets() {
    let fixture = Fixture::new();
    fixture.write("notes.txt", "abcdefghij\n");

    let result = fixture
        .call(
            &ReadMany,
            json!({"paths": ["notes.txt"], "max_total_chars": 5}),
        )
        .await
        .unwrap();
    assert_eq!(result["files"][0]["content"], "1|abc");
    assert_eq!(result["files"][0]["truncated"], true);
    assert_eq!(result["total_chars"], 5);
    assert_eq!(result["max_total_chars"], 5);
    assert_eq!(result["truncated"], true);

    let paths: Vec<String> = (0..21).map(|index| format!("missing{index}.txt")).collect();
    let result = fixture
        .call(&ReadMany, json!({"paths": paths}))
        .await
        .unwrap();
    assert_eq!(result["requested_count"], 21);
    assert_eq!(result["processed_count"], 20);
    assert_eq!(result["truncated"], true);
}

#[tokio::test]
async fn search_files_finds_files_by_name() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("agent/models.py", "");
    fixture.write("tools/workspace.py", "");

    let result = fixture
        .call(&SearchFiles, json!({"pattern": "models.py"}))
        .await
        .unwrap();

    assert_eq!(
        result,
        json!({
            "target": "files",
            "pattern": "models.py",
            "path": ".",
            "files": ["agent/models.py"],
            "offset": 0,
            "limit": 50,
            "total_count": 1,
            "total_count_is_exact": true,
            "truncated": false,
        })
    );
}

#[tokio::test]
async fn search_files_pages_within_a_path() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("src/a.py", "needle one\n");
    fixture.write("src/b.py", "needle two\n");
    fixture.write("src/c.py", "needle three\n");
    fixture.write("notes.py", "needle ignored\n");

    let files = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "*.py", "path": "src", "limit": 1}),
        )
        .await
        .unwrap();
    assert_eq!(files["path"], "src");
    assert_eq!(files["files"], json!(["src/a.py"]));
    assert_eq!(files["total_count"], 2);
    assert_eq!(files["total_count_is_exact"], false);
    assert_eq!(files["truncated"], true);
    assert_eq!(files["next_offset"], 1);

    let second = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "*.py", "path": "src", "offset": 1, "limit": 5}),
        )
        .await
        .unwrap();
    assert_eq!(second["files"], json!(["src/b.py", "src/c.py"]));
    assert_eq!(second["total_count"], 3);
    assert_eq!(second["total_count_is_exact"], true);
    assert_eq!(second["truncated"], false);
    assert!(second.get("next_offset").is_none());

    let content = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content", "path": "src", "limit": 1}),
        )
        .await
        .unwrap();
    assert_eq!(
        content["matches"],
        json!([{"path": "src/a.py", "line": 1, "text": "needle one"}])
    );
    assert_eq!(content["total_count"], 2);
    assert_eq!(content["truncated"], true);
}

#[tokio::test]
async fn search_files_finds_content_matches() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write(
        "tools/registry.py",
        "import x\r\nclass ToolRegistry:\n    pass\n",
    );

    let result = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "ToolRegistry", "target": "content"}),
        )
        .await
        .unwrap();

    assert_eq!(
        result,
        json!({
            "target": "content",
            "pattern": "ToolRegistry",
            "path": ".",
            "matches": [{"path": "tools/registry.py", "line": 2, "text": "class ToolRegistry:"}],
            "offset": 0,
            "limit": 50,
            "total_count": 1,
            "total_count_is_exact": true,
            "truncated": false,
        })
    );

    let none = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "absent", "target": "content"}),
        )
        .await
        .unwrap();
    assert_eq!(none["matches"], json!([]));
    assert_eq!(none["total_count"], 0);
}

#[tokio::test]
async fn search_files_bounds_long_matching_lines() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("minified.txt", format!("needle{}\n", "x".repeat(8_000)));

    let result = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content"}),
        )
        .await
        .unwrap();

    let found = &result["matches"][0];
    assert_eq!(found["text"].as_str().unwrap().chars().count(), 4_000);
    assert_eq!(found["text_truncated"], true);

    // ripgrep measures long lines in bytes but cuts its preview by
    // characters, so a line of four-byte characters makes a record of about
    // 64 KB.
    fixture.write("emoji.txt", format!("crab{}\n", "🦀".repeat(17_000)));
    let result = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "crab", "target": "content", "path": "emoji.txt"}),
        )
        .await
        .unwrap();
    let found = &result["matches"][0];
    let text = found["text"].as_str().unwrap();
    assert!(text.starts_with("crab🦀"));
    assert_eq!(text.chars().count(), 4_000);
    assert!(
        !text.contains("omitted"),
        "ripgrep's marker is replaced by the flag"
    );
    assert_eq!(found["text_truncated"], true);

    // A multibyte line over 4,000 bytes but under 4,000 characters is whole.
    fixture.write("accents.txt", format!("café{}\n", "é".repeat(2_500)));
    let result = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "café", "target": "content", "path": "accents.txt"}),
        )
        .await
        .unwrap();
    let found = &result["matches"][0];
    let text = found["text"].as_str().unwrap();
    assert_eq!(text.chars().count(), 2_504);
    assert!(text.ends_with('é'));
    assert!(found.get("text_truncated").is_none());
}

#[tokio::test]
async fn content_searches_stay_within_an_output_budget() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    let line = format!("needle {}\n", "x".repeat(3_900));
    fixture.write("wide.txt", line.repeat(60));

    let result = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content", "limit": 60}),
        )
        .await
        .unwrap();

    let returned = result["matches"].as_array().unwrap().len();
    assert!((20..30).contains(&returned), "{returned} matches");
    let text_chars: usize = result["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|found| found["text"].as_str().unwrap().chars().count())
        .sum();
    assert!(text_chars <= 100_000);
    assert_eq!(result["truncated"], true);
    assert_eq!(result["next_offset"], returned);
    assert_eq!(result["total_count"], 60);
    assert_eq!(result["total_count_is_exact"], true);

    let next = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content", "limit": 60, "offset": returned}),
        )
        .await
        .unwrap();
    assert_eq!(next["matches"][0]["line"], returned + 1);
}

#[tokio::test]
async fn binary_files_are_skipped_not_fatal() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("data.bin", b"needle\0binary\n");
    fixture.write("text.txt", "needle\n");

    let direct = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content", "path": "data.bin"}),
        )
        .await
        .unwrap();
    assert_eq!(direct["matches"], json!([]));

    let walked = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content"}),
        )
        .await
        .unwrap();
    assert_eq!(
        walked["matches"],
        json!([{"path": "text.txt", "line": 1, "text": "needle"}])
    );
}

#[cfg(unix)]
#[tokio::test]
async fn unreadable_directories_do_not_hide_other_results() {
    use std::os::unix::fs::PermissionsExt;

    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("locked/hidden.txt", "needle\n");
    fixture.write("open.txt", "needle\n");
    let locked = fixture.workspace.join("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    // Permissions do not apply to the superuser.
    let enforced = fs::read_dir(&locked).is_err();

    let content = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "needle", "target": "content"}),
        )
        .await;
    let files = fixture
        .call(&SearchFiles, json!({"pattern": "*.txt"}))
        .await;
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();

    if enforced {
        let content = content.unwrap();
        assert_eq!(
            content["matches"],
            json!([{"path": "open.txt", "line": 1, "text": "needle"}])
        );
        assert_eq!(files.unwrap()["files"], json!(["open.txt"]));
    }
}

#[tokio::test]
async fn search_files_respects_ignore_files_and_single_file_paths() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    // The workspace is not a git repository; `.gitignore` applies anyway.
    fixture.write(".gitignore", "node_modules/\n");
    fixture.write(".ignore", "build/\n");
    fixture.write("kept.txt", "");
    fixture.write("build/out.txt", "");
    fixture.write("node_modules/dep/readme.txt", "");
    fixture.write("src/lib.rs", "");

    let result = fixture
        .call(&SearchFiles, json!({"pattern": "*.txt"}))
        .await
        .unwrap();
    assert_eq!(result["files"], json!(["kept.txt"]));

    let file = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "lib", "path": "src/lib.rs"}),
        )
        .await
        .unwrap();
    assert_eq!(file["path"], "src/lib.rs");
    assert_eq!(file["files"], json!(["src/lib.rs"]));
    let other = fixture
        .call(
            &SearchFiles,
            json!({"pattern": "*.py", "path": "src/lib.rs"}),
        )
        .await
        .unwrap();
    assert_eq!(other["files"], json!([]));
    assert_eq!(other["total_count"], 0);
}

#[tokio::test]
async fn search_files_reports_bad_paths_and_patterns() {
    assert!(ripgrep_installed(), "search tests need ripgrep (rg)");
    let fixture = Fixture::new();
    fixture.write("a.txt", "a\n");
    let outside = fixture.outside.display().to_string();

    let error = fixture
        .call(&SearchFiles, json!({"pattern": "*", "path": outside}))
        .await
        .unwrap_err();
    assert_eq!(error, format!("Path escapes workspace: {outside}"));

    let error = fixture
        .call(&SearchFiles, json!({"pattern": "*", "path": "nope"}))
        .await
        .unwrap_err();
    assert_eq!(error, "Path not found: nope");

    let error = fixture
        .call(&SearchFiles, json!({"pattern": "(", "target": "content"}))
        .await
        .unwrap_err();
    assert!(error.starts_with("ripgrep failed: "), "{error}");
    assert!(error.contains("regex parse error"), "{error}");

    let error = fixture
        .call(&SearchFiles, json!({"pattern": "a", "target": "names"}))
        .await
        .unwrap_err();
    assert!(error.starts_with("Invalid arguments for search_files: unknown variant `names`"));
}

#[test]
fn the_default_registry_advertises_the_built_in_tools() {
    let registry = default_registry();
    assert_eq!(
        registry.names().collect::<Vec<_>>(),
        ["read_file", "read_many", "search_files"]
    );
    let definitions = registry.definitions();
    assert_eq!(
        Value::Object(definitions[0].parameters.clone()),
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        })
    );
    assert_eq!(
        definitions[1].parameters["properties"]["paths"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    assert_eq!(
        definitions[2].parameters["properties"]["target"],
        json!({"type": "string", "enum": ["files", "content"]})
    );
}

#[tokio::test]
async fn tools_run_through_the_registry_with_correlated_results() {
    let fixture = Fixture::new();
    fixture.write("notes.txt", "hello\n");
    let registry = default_registry();
    let context = ToolContext::new(Path::new(&fixture.workspace), CancellationToken::new());
    let mut arguments = JsonObject::new();
    arguments.insert("path".into(), "notes.txt".into());
    let call = ToolCall::new("read_file", arguments);

    let result = registry.execute(&call, &context).await.unwrap();

    assert!(result.ok);
    assert_eq!(result.tool_call_id, call.id);
    let output: Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(output["content"], "1|hello");
    assert_eq!(result.metadata["permission"], "read");
}
