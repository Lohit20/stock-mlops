"""
Tests for src/versioning/dvc_ops.py

All subprocess/git/dvc calls are mocked — no real DVC or git needed.
"""

import os
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from src.versioning.dvc_ops import (
    _run,
    _ensure_git_config,
    _git_commit,
    add_to_dvc,
    get_current_hash,
    list_versions,
    snapshot_raw_data,
    snapshot_processed_data,
    snapshot_features,
)


# ── _run ──────────────────────────────────────────────────────────────────────

@patch("subprocess.run")
def test_run_executes_command(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    _run(["echo", "hello"])
    mock_run.assert_called_once_with(
        ["echo", "hello"], check=True, capture_output=True, text=True
    )


@patch("subprocess.run")
def test_run_with_check_false(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    result = _run(["false"], check=False)
    assert result.returncode == 1


# ── _ensure_git_config ────────────────────────────────────────────────────────

@patch("src.versioning.dvc_ops._run")
def test_ensure_git_config_sets_identity_when_missing(mock_run):
    mock_run.return_value = MagicMock(stdout="", returncode=0)
    _ensure_git_config()
    calls = [str(c) for c in mock_run.call_args_list]
    assert any("user.name" in c for c in calls)
    assert any("user.email" in c for c in calls)


@patch("src.versioning.dvc_ops._run")
def test_ensure_git_config_skips_when_already_set(mock_run):
    check_call = MagicMock(stdout="mlops-pipeline\n", returncode=0)
    mock_run.return_value = check_call
    _ensure_git_config()
    # Only the check call should happen, no config set calls
    assert mock_run.call_count == 1


# ── _git_commit ───────────────────────────────────────────────────────────────

@patch("src.versioning.dvc_ops._ensure_git_config")
@patch("src.versioning.dvc_ops._run")
def test_git_commit_only_commits_when_staged(mock_run, mock_config):
    # Simulate staged changes
    mock_run.return_value = MagicMock(stdout="M data/raw.dvc\n", returncode=0)
    _git_commit("test: snapshot")
    commit_calls = [c for c in mock_run.call_args_list if "commit" in str(c)]
    assert len(commit_calls) == 1


@patch("src.versioning.dvc_ops._ensure_git_config")
@patch("src.versioning.dvc_ops._run")
def test_git_commit_skips_when_nothing_staged(mock_run, mock_config):
    mock_run.return_value = MagicMock(stdout="", returncode=0)
    _git_commit("test: no changes")
    commit_calls = [c for c in mock_run.call_args_list if "commit" in str(c)]
    assert len(commit_calls) == 0


# ── add_to_dvc ────────────────────────────────────────────────────────────────

@patch("src.versioning.dvc_ops._run")
def test_add_to_dvc_calls_dvc_add(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="")
    add_to_dvc("data/raw")
    mock_run.assert_called_once_with(["dvc", "add", "data/raw"])


@patch("src.versioning.dvc_ops._run")
def test_add_to_dvc_returns_hash_from_dvc_file(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="")

    # Write a real .dvc file in tmp_path so there's no open() mock recursion
    dvc_file = tmp_path / "raw.dvc"
    dvc_file.write_text("outs:\n  - md5: abc123\n    path: raw\n")

    with patch("src.versioning.dvc_ops.os.path.exists", return_value=True), \
         patch("builtins.open", return_value=dvc_file.open()):
        result = add_to_dvc(str(tmp_path / "raw"))

    assert isinstance(result, str)


# ── get_current_hash ──────────────────────────────────────────────────────────

def test_get_current_hash_returns_not_tracked_when_no_dvc_file():
    with patch("src.versioning.dvc_ops.os.path.exists", return_value=False):
        result = get_current_hash("data/raw")
    assert result == "not-tracked"


def test_get_current_hash_parses_dvc_file(tmp_path):
    # Write a real .dvc metafile — avoids recursive open() mock
    dvc_file = tmp_path / "raw.dvc"
    dvc_file.write_text("outs:\n  - md5: deadbeef\n    path: raw\n")

    tracked_path = str(tmp_path / "raw")
    result = get_current_hash(tracked_path)   # will look for <tracked_path>.dvc
    assert result == "deadbeef"


# ── list_versions ─────────────────────────────────────────────────────────────

@patch("src.versioning.dvc_ops._run")
def test_list_versions_parses_git_log(mock_run):
    mock_run.return_value = MagicMock(
        stdout="abc123|2024-01-01 12:00:00 +0000|data: raw snapshot 2024-01-01\n"
               "def456|2024-01-02 12:00:00 +0000|data: raw snapshot 2024-01-02\n",
        returncode=0,
    )
    versions = list_versions(n=2)
    assert len(versions) == 2
    assert versions[0]["sha"] == "abc123"
    assert versions[1]["sha"] == "def456"


@patch("src.versioning.dvc_ops._run")
def test_list_versions_returns_empty_on_no_history(mock_run):
    mock_run.return_value = MagicMock(stdout="", returncode=0)
    assert list_versions() == []


# ── snapshot_raw_data ─────────────────────────────────────────────────────────

@patch("src.versioning.dvc_ops.push_to_remote", return_value=True)
@patch("src.versioning.dvc_ops._git_commit")
@patch("src.versioning.dvc_ops.add_to_dvc", return_value="abc123hash")
@patch("src.versioning.dvc_ops._run")
def test_snapshot_raw_data_returns_info(mock_run, mock_add, mock_commit, mock_push):
    mock_run.return_value = MagicMock(stdout="deadbeef1234\n", returncode=0)
    info = snapshot_raw_data()
    assert info["hash"] == "abc123hash"
    assert "commit" in info
    assert info["path"] == os.getenv("RAW_DATA_PATH", "data/raw")


@patch("src.versioning.dvc_ops.push_to_remote", return_value=False)
@patch("src.versioning.dvc_ops._git_commit")
@patch("src.versioning.dvc_ops.add_to_dvc", return_value="abc123")
@patch("src.versioning.dvc_ops._run")
def test_snapshot_raw_data_continues_when_push_fails(mock_run, mock_add, mock_commit, mock_push):
    mock_run.return_value = MagicMock(stdout="abc\n", returncode=0)
    # Should not raise even if push fails
    info = snapshot_raw_data()
    assert info is not None
