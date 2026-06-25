"""End-to-end tests for the pybench CLI (run / update / show)."""

import subprocess
from pathlib import Path

from click.testing import CliRunner

from pybench import cli
from pybench.cli import main
from pybench.store import read_baselines

SCALAR = (
    "import random\n"
    "def bench_a(seed, *, n_seeds=8):\n"
    "    r = random.Random(seed)\n"
    "    return 0.9 + r.gauss(0, 0.01)\n"
)
# Same seeds, but a desynced RNG stream and lower mean — a real regression.
REGRESS = (
    "import random\n"
    "def bench_a(seed, *, n_seeds=8):\n"
    "    r = random.Random(seed)\n"
    "    r.random()\n"
    "    return 0.6 + r.gauss(0, 0.01)\n"
)


def _make(body, name="bench_a.py"):
    Path("benchmarks").mkdir(exist_ok=True)
    (Path("benchmarks") / name).write_text(body)


def _git(*args):
    subprocess.run(["git", *args], check=True, capture_output=True, text=True)


def _init_repo():
    _git("init")
    _git("config", "user.email", "t@t.co")
    _git("config", "user.name", "t")
    _git("config", "commit.gpgsign", "false")


def _commit(msg):
    _git("add", "-A")
    _git("commit", "-m", msg)


def _dirty_repo_with_new_bench():
    """A repo with one commit (so HEAD exists) and an untracked benchmark."""
    _init_repo()
    Path("seed.txt").write_text("x")
    _commit("init")
    _make(SCALAR)  # untracked -> working tree is dirty


def test_new_then_pass():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        first = runner.invoke(main, ["benchmarks"])
        assert first.exit_code == 0
        assert "NEW" in first.output
        assert Path(".pybench/baselines.jsonl").exists()
        second = runner.invoke(main, ["benchmarks"])
        assert second.exit_code == 0
        assert "PASS" in second.output


def test_regression_fails():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        runner.invoke(main, ["benchmarks"])
        _make(REGRESS)
        result = runner.invoke(main, ["benchmarks", "-v"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


def test_discovery_error_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["nope"])
        assert result.exit_code == 2
        assert "error:" in result.output


def test_run_unreachable_alpha_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # 1/2**4 = 0.0625 >= alpha=0.05: no regression could ever fail the gate.
        _make("def bench_a(seed, *, n_seeds=4):\n    return 0.9\n")
        result = runner.invoke(main, ["benchmarks"])
        assert result.exit_code == 2
        assert "can never flag" in result.output
        assert not Path(".pybench/baselines.jsonl").exists()


def test_update_unreachable_alpha_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make("def bench_a(seed, *, n_seeds=4):\n    return 0.9\n")
        result = runner.invoke(main, ["update", "benchmarks", "--yes"])
        assert result.exit_code == 2
        assert "can never flag" in result.output


def test_metric_mismatch_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make('def bench_a(seed, *, n_seeds=5):\n    return {"a": 1.0}\n')
        runner.invoke(main, ["benchmarks"])
        _make('def bench_a(seed, *, n_seeds=5):\n    return {"b": 1.0}\n')
        result = runner.invoke(main, ["benchmarks"])
        assert result.exit_code == 2
        assert "error:" in result.output


def test_bench_filter_and_overrides():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        _make(
            "import random\n"
            "def bench_b(seed, *, n_seeds=8):\n"
            "    r = random.Random(seed)\n"
            "    return 0.5 + r.gauss(0, 0.01)\n",
            name="bench_b.py",
        )
        result = runner.invoke(
            main,
            [
                "benchmarks",
                "--bench",
                "bench_a",
                "--alpha",
                "0.05",
                "--min-effect",
                "0.01",
                "-v",
            ],
        )
        assert result.exit_code == 0
        assert "bench_a" in result.output
        assert "bench_b" not in result.output


# --- dirty-tree prompt on run --------------------------------------------


def test_run_dirty_prompt_declined():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _dirty_repo_with_new_bench()
        result = runner.invoke(main, ["benchmarks"], input="n\n")
        assert result.exit_code == 0
        assert "dirty" in result.output
        assert "Aborted" in result.output
        assert not Path(".pybench/baselines.jsonl").exists()


def test_run_dirty_prompt_accepted():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _dirty_repo_with_new_bench()
        result = runner.invoke(main, ["benchmarks"], input="y\n")
        assert result.exit_code == 0
        assert "NEW" in result.output
        assert Path(".pybench/baselines.jsonl").exists()


def test_run_dirty_yes_skips_prompt():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _dirty_repo_with_new_bench()
        result = runner.invoke(main, ["benchmarks", "--yes"])
        assert result.exit_code == 0
        assert "dirty" in result.output
        assert "NEW" in result.output


# --- update ---------------------------------------------------------------


def test_update_yes_writes_dirty():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _dirty_repo_with_new_bench()
        result = runner.invoke(main, ["update", "benchmarks", "--yes"])
        assert result.exit_code == 0
        assert "rewritten" in result.output
        assert read_baselines(Path(".pybench/baselines.jsonl"))["bench_a"].git_dirty


def test_update_clean_confirm_accepted_with_n_seeds():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _init_repo()
        _make(SCALAR)
        _commit("bench")  # clean tree
        result = runner.invoke(
            main, ["update", "benchmarks", "--n-seeds", "6"], input="y\n"
        )
        assert result.exit_code == 0
        assert "Overwrite baseline" in result.output
        recs = read_baselines(Path(".pybench/baselines.jsonl"))
        assert len(recs["bench_a"].seeds) == 6


def test_update_declined():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        result = runner.invoke(main, ["update", "benchmarks"], input="n\n")
        assert result.exit_code == 0
        assert "unchanged" in result.output
        assert not Path(".pybench/baselines.jsonl").exists()


def test_update_preserves_other_benchmarks():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        _make(SCALAR.replace("bench_a", "bench_b"), name="bench_b.py")
        runner.invoke(main, ["update", "benchmarks", "--yes"])
        result = runner.invoke(
            main, ["update", "benchmarks", "--bench", "bench_a", "--yes"]
        )
        assert result.exit_code == 0
        recs = read_baselines(Path(".pybench/baselines.jsonl"))
        assert set(recs) == {"bench_a", "bench_b"}


def test_update_discovery_error_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["update", "nope", "--yes"])
        assert result.exit_code == 2
        assert "error:" in result.output


def test_update_run_shape_error_exits_2():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(
            "import itertools\n"
            "_c = itertools.count()\n"
            "def bench_a(seed, *, n_seeds=8):\n"
            '    return {"a": 1.0} if next(_c) == 0 else {"b": 1.0}\n'
        )
        result = runner.invoke(main, ["update", "benchmarks", "--yes"])
        assert result.exit_code == 2
        assert "error:" in result.output


# --- show -----------------------------------------------------------------


def test_show_default_and_filter():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _make(SCALAR)
        _make(SCALAR.replace("bench_a", "bench_b"), name="bench_b.py")
        runner.invoke(main, ["update", "benchmarks", "--yes"])
        full = runner.invoke(main, ["show"])
        assert full.exit_code == 0
        assert "bench_a" in full.output and "bench_b" in full.output
        filtered = runner.invoke(main, ["show", "--bench", "bench_a"])
        assert "bench_a" in filtered.output
        assert "bench_b" not in filtered.output


def test_show_history():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _init_repo()
        _make(SCALAR)
        _make(SCALAR.replace("bench_a", "bench_b"), name="bench_b.py")
        runner.invoke(main, ["update", "benchmarks", "--yes"])
        _commit("baseline v1")
        runner.invoke(main, ["update", "benchmarks", "--yes"])
        _commit("baseline v2")
        # --bench filters bench_b out of the per-commit records.
        result = runner.invoke(main, ["show", "--history", "--bench", "bench_a"])
        assert result.exit_code == 0
        assert "bench_b" not in result.output
        assert result.output.count("score@0") == 2  # one line per commit


def test_show_history_no_git():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["show", "--history"])
        assert result.exit_code == 0
        assert "not a git repository" in result.output


def test_show_history_skips_unreadable_commit(monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _init_repo()
        _make(SCALAR)
        runner.invoke(main, ["update", "benchmarks", "--yes"])
        _commit("baseline")
        monkeypatch.setattr(cli, "file_at_commit", lambda *a, **k: None)
        result = runner.invoke(main, ["show", "--history"])
        assert result.exit_code == 0
        assert "No baseline history found." in result.output
