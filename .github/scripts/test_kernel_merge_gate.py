import io
import json
from pathlib import Path
import urllib.error

import pytest

import dispatch
import pr_comment_kernel_bot as bot


API = "https://api.github.com/repos/owner/repo"
HEAD = "a" * 40
CONTEXT = "Build (macOS) / sample"


def test_latest_statuses_are_paginated_and_keep_latest_context(monkeypatch):
    calls = []
    first = [{"context": CONTEXT, "state": "pending"}] + [
        {"context": f"other-{i}", "state": "success"} for i in range(99)
    ]
    second = [
        {"context": CONTEXT, "state": "success"},
        {"context": "Build / sample", "state": "success"},
    ]

    def request(url, token):
        calls.append(url)
        assert token == "token"
        batch = first if len(calls) == 1 else second
        return 200, json.dumps({"statuses": batch})

    monkeypatch.setattr(bot, "github_api_request", request)
    statuses = bot.latest_commit_statuses(API, "token", HEAD)

    assert statuses[CONTEXT]["state"] == "pending"
    assert statuses["Build / sample"]["state"] == "success"
    assert calls == [
        f"{API}/commits/{HEAD}/status?per_page=100&page={page}" for page in (1, 2)
    ]


@pytest.mark.parametrize("state", [None, "pending", "failure", "error", "success"])
def test_preflight_requires_success_for_current_head(monkeypatch, state):
    checked = []

    def read_backends(kernel, ref):
        checked.append((kernel, ref))
        return ["metal"]

    monkeypatch.setattr(dispatch, "read_backends", read_backends)
    statuses = {"Unrelated check": {"state": "failure"}}
    if state:
        statuses[CONTEXT] = {"state": state}
    monkeypatch.setattr(bot, "latest_commit_statuses", lambda *args: statuses)

    failures = bot.preflight_kernel_builds(API, "token", HEAD, ["sample"])

    assert checked == [("sample", HEAD)]
    assert failures == (
        [] if state == "success" else [f"`{CONTEXT}`: {state or 'missing'}."]
    )


def test_preflight_requires_every_kernel_and_supported_platform(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "read_backends",
        lambda kernel, ref: ["cuda"] if kernel == "relu" else ["metal"],
    )
    monkeypatch.setattr(
        bot,
        "latest_commit_statuses",
        lambda *args: {"Build / relu": {"state": "success"}},
    )

    assert bot.preflight_kernel_builds(API, "token", HEAD, ["relu", "sample"]) == [
        "`Build (Windows) / relu`: missing.",
        f"`{CONTEXT}`: missing.",
    ]


def test_preflight_obeys_windows_allowlist(monkeypatch):
    monkeypatch.setattr(dispatch, "read_backends", lambda *args: ["cuda"])
    monkeypatch.setattr(
        bot,
        "latest_commit_statuses",
        lambda *args: {"Build / flash-attn2": {"state": "success"}},
    )
    assert bot.preflight_kernel_builds(API, "token", HEAD, ["flash-attn2"]) == []


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        ValueError("invalid JSON"),
        KeyError("statuses"),
    ],
)
def test_preflight_fails_closed_on_status_lookup_errors(monkeypatch, error):
    def statuses(*args):
        raise error

    monkeypatch.setattr(bot, "latest_commit_statuses", statuses)
    assert (
        "Could not read"
        in bot.preflight_kernel_builds(API, "token", HEAD, ["sample"])[0]
    )


def test_preflight_rejects_missing_head_without_querying_statuses(monkeypatch):
    def unexpected(*args):
        raise AssertionError("must not query a branch instead of a commit")

    monkeypatch.setattr(bot, "latest_commit_statuses", unexpected)
    assert "head commit" in bot.preflight_kernel_builds(API, "token", "", ["sample"])[0]


@pytest.fixture
def merge_command(monkeypatch):
    ctx = {
        "token": "token",
        "repository": "owner/repo",
        "comment": "/kernel-bot merge-and-upload sample",
        "comment_id": 7,
        "issue_number": 42,
        "default_branch": "main",
        "commenter": "maintainer",
    }
    pull_request = {"head": {"sha": HEAD}, "state": "open", "merged": False}
    calls, comments = [], []
    state = {"build": "success", "head_changed": False}

    monkeypatch.setattr(bot, "_resolve_context_from_env", lambda: ctx)
    monkeypatch.setattr(bot, "try_post_issue_comment_reaction", lambda *a, **k: True)
    monkeypatch.setattr(bot, "get_user_permission", lambda *a, **k: "admin")
    monkeypatch.setattr(bot, "get_pull_request", lambda *a, **k: pull_request)
    monkeypatch.setattr(bot, "preflight_external_uploads", lambda *a, **k: [])
    monkeypatch.setattr(dispatch, "read_backends", lambda *a, **k: ["metal"])
    monkeypatch.setattr(bot, "try_create_issue_comment", lambda *a, **k: {"id": 8})
    monkeypatch.setattr(bot, "resolve_dispatch_run_urls", lambda *a, **k: None)

    def comment(api_base, token, issue_number, message, **kwargs):
        comments.append(message)
        return True

    monkeypatch.setattr(bot, "try_send_issue_comment", comment)

    def request(url, token, method="GET", data=None):
        if url == f"{API}/commits/{HEAD}/status?per_page=100&page=1":
            assert method == "GET"
            calls.append("statuses")
            statuses = (
                [{"context": CONTEXT, "state": state["build"]}]
                if state["build"]
                else []
            )
            return 200, json.dumps({"statuses": statuses})
        assert (url, method) == (f"{API}/pulls/42/merge", "PUT")
        assert data == {"merge_method": "squash", "sha": HEAD}
        calls.append("merge")
        if state["head_changed"]:
            raise urllib.error.HTTPError(
                url, 409, "Head changed", {}, io.BytesIO(b'{"message":"Head changed"}')
            )
        return 200, json.dumps({"merged": True})

    monkeypatch.setattr(bot, "github_api_request", request)

    def release(kernel_name, **kwargs):
        calls.append("dispatch")
        return dispatch.DispatchResult(kernel_name=kernel_name)

    monkeypatch.setattr(bot, "dispatch", release)
    return ctx, pull_request, state, calls, comments


@pytest.mark.parametrize("build", [None, "pending", "failure", "error"])
def test_merge_command_stops_before_merge_or_release(merge_command, build):
    _, _, state, calls, comments = merge_command
    state["build"] = build

    assert bot.main() == 1
    assert calls == ["statuses"]
    assert "no merge or release was started" in comments[-1]
    assert "/kernel-bot build sample" in comments[-1]
    assert "/kernel-bot build-and-stage sample" in comments[-1]


def test_merge_command_merges_checked_commit_then_releases(merge_command):
    _, _, _, calls, _ = merge_command

    assert bot.main() == 0
    assert calls == ["statuses", "merge", "dispatch"]


def test_new_push_during_preflight_does_not_merge_or_release(merge_command):
    _, _, state, calls, comments = merge_command
    state["head_changed"] = True

    assert bot.main() == 1
    assert calls == ["statuses", "merge"]
    assert "Failed to merge" in comments[-1]


def test_backend_filter_cannot_skip_required_platforms(merge_command, monkeypatch):
    ctx, _, _, calls, comments = merge_command
    ctx["comment"] = "/kernel-bot merge-and-upload sample[metal]"
    monkeypatch.setattr(dispatch, "read_backends", lambda *a: ["cpu", "metal"])

    assert bot.main() == 1
    assert calls == ["statuses"]
    assert "`Build / sample`: missing" in comments[-1]


def test_already_merged_pr_can_retry_failed_release(merge_command):
    _, pr, state, calls, _ = merge_command
    pr.update(merged=True, state="closed")
    state["build"] = "failure"

    assert bot.main() == 0
    assert calls == ["dispatch"]


@pytest.mark.parametrize("command", ["build", "build-and-stage", "release"])
def test_non_merging_commands_do_not_require_previous_builds(merge_command, command):
    ctx, _, state, calls, _ = merge_command
    ctx["comment"] = f"/kernel-bot {command} sample"
    state["build"] = None

    assert bot.main() == 0
    assert calls == ["dispatch"]


def test_comment_workflow_can_read_and_queue_build_statuses():
    workflow = Path(".github/workflows/pr-comment-build.yaml").read_text()
    assert "  statuses: write\n" in workflow
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in workflow
