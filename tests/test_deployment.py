"""The per-caller allowance and the shared daily cap are counted in memory,
so they are only exact if the service runs one worker process: with several,
each keeps its own counter and the numbers a caller sees jump around (seen
live: 22, 23, 21, 18 across four single requests). Threads are fine, the
counters are lock-protected."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _start_command(text: str) -> str:
    return next(line for line in text.splitlines() if "gunicorn" in line)


def test_procfile_runs_a_single_worker_with_threads():
    command = _start_command((ROOT / "Procfile").read_text())

    assert "--workers 1" in command
    assert "--threads" in command


def test_render_start_command_matches_the_procfile():
    procfile = _start_command((ROOT / "Procfile").read_text())
    render = _start_command((ROOT / "render.yaml").read_text())

    assert "--workers 1" in render and "--threads" in render
    assert procfile.split("gunicorn", 1)[1] == render.split("gunicorn", 1)[1]


def test_trusted_proxy_hops_matches_what_render_really_adds():
    """Measured live (2026-09-27): a request with no client header already
    arrives with three X-Forwarded-For entries, so the caller's address is
    the third from the right. With 1, the app took Render's own last proxy,
    whose address alternates, and split every caller across buckets."""
    text = (ROOT / "render.yaml").read_text()
    lines = text.splitlines()
    index = next(i for i, l in enumerate(lines) if "key: TRUSTED_PROXY_HOPS" in l)

    assert 'value: "3"' in lines[index + 1]
