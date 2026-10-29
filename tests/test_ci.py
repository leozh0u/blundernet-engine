import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_exhausted_checkpoint_upload_is_a_failed_run(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"offline fixture")
    for name, body in (("gh", "exit 1"), ("sleep", "exit 0")):
        command = tmp_path / name
        command.write_text(f"#!/bin/sh\n{body}\n")
        command.chmod(0o755)
    run = subprocess.run(
        ["bash", str(ROOT / "scripts" / "ci_release.sh")],
        cwd=tmp_path, env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
    )
    assert run.returncode != 0


@pytest.mark.parametrize("workflow", ["train.yml", "backup.yml"])
def test_checkpoint_restore_failure_stops_the_workflow(tmp_path, workflow):
    source = (ROOT / ".github" / "workflows" / workflow).read_text()
    step = source.split("- name: Restore latest checkpoint\n", 1)[1].split("\n      - ", 1)[0]
    script = textwrap.dedent(step.split("run: |\n", 1)[1])
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\nexit 1\n")
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    run = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env, capture_output=True)
    assert run.returncode != 0
    assert not (tmp_path / "checkpoint" / "model.pt").exists()


def test_push_does_not_resolve_conflicting_training_state_automatically(tmp_path):
    git = tmp_path / "git"
    git.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'case "$1" in\n'
        '  pull) case "$*" in *"-X ours"*) exit 0;; *) exit 1;; esac;;\n'
        'esac\nexit 0\n'
    )
    git.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CALLS": str(calls)}
    run = subprocess.run(["bash", str(ROOT / "scripts" / "ci_push.sh")],
                         cwd=tmp_path, env=env, capture_output=True)
    assert run.returncode != 0
    assert "push" not in calls.read_text().splitlines()
    assert calls.read_text().count("pull --rebase") == 5
