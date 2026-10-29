import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_puzzles", Path(__file__).resolve().parents[1] / "scripts" / "build_puzzles.py"
)
build_puzzles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_puzzles)


def test_incomplete_download_cannot_replace_the_fixed_eval_suite(tmp_path, monkeypatch):
    class EmptyDownload:
        stdout = io.BytesIO(b"")

        def terminate(self):
            pass

    target = tmp_path / "puzzles.csv"
    target.write_text("existing fixed evaluation suite\n")
    monkeypatch.setattr(build_puzzles, "OUT", target)
    monkeypatch.setattr(build_puzzles.subprocess, "Popen", lambda *a, **kw: EmptyDownload())
    with pytest.raises(RuntimeError, match="incomplete"):
        build_puzzles.main()
    assert target.read_text() == "existing fixed evaluation suite\n"
