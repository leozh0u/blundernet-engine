#!/usr/bin/env python3
"""One scheduled training run: ingest -> train -> eval -> log, committing per stage.

Usage: python scripts/pipeline.py [--no-commit] [--chart]
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

METRICS_DIR = Path("metrics")
HISTORY = METRICS_DIR / "history.csv"
LATEST = METRICS_DIR / "latest.json"
FIELDS = [
    "timestamp", "steps", "samples_seen", "games", "positions",
    "loss", "policy_loss", "value_loss", "top1", "top3",
    "puzzle_overall", "puzzle_800-1200", "puzzle_1200-1600",
    "puzzle_1600-2000", "puzzle_2000-2400", "puzzle_2400-+",
]


def git_commit(message: str, no_commit: bool) -> None:
    if no_commit:
        print(f"[skip commit] {message}")
        return
    subprocess.run(["git", "add", "-A"], check=True)
    r = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
    print(r.stdout or r.stderr)


def append_history(row: dict) -> None:
    METRICS_DIR.mkdir(exist_ok=True)
    exists = HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def make_chart() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(HISTORY.open()))
    if len(rows) < 2:
        return
    steps = [int(r["steps"]) for r in rows]

    def col(name):
        return [float(r[name]) if r.get(name) else float("nan") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(steps, col("loss"), label="total")
    axes[0].plot(steps, col("policy_loss"), label="policy")
    axes[0].set_title("training loss"); axes[0].set_xlabel("optimizer steps"); axes[0].legend()

    axes[1].plot(steps, [100 * v for v in col("top1")], label="top-1")
    axes[1].plot(steps, [100 * v for v in col("top3")], label="top-3")
    axes[1].set_title("held-out move prediction (%)")
    axes[1].set_xlabel("optimizer steps"); axes[1].legend()

    for b in ("puzzle_800-1200", "puzzle_1200-1600", "puzzle_1600-2000",
              "puzzle_2000-2400", "puzzle_2400-+"):
        axes[2].plot(steps, [100 * v for v in col(b)], label=b.replace("puzzle_", ""))
    axes[2].plot(steps, [100 * v for v in col("puzzle_overall")], "k--", lw=2, label="overall")
    axes[2].set_title("tactics puzzle accuracy by rating (%)")
    axes[2].set_xlabel("optimizer steps"); axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(METRICS_DIR / "curve.png", dpi=110)
    print("chart updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--commits", type=int, default=None,
                    help="target number of commits this run (training volume "
                         "is unaffected; work is squashed to fit). Default: "
                         "one commit per stage, as before.")
    args = ap.parse_args()

    from blundernet.data import gather_batch
    from blundernet.evaluate import move_accuracy
    from blundernet.puzzles import evaluate_puzzles
    from blundernet.train import load_model, save_model, train_on_batch

    model, opt, meta = load_model()
    now = dt.datetime.now(dt.timezone.utc)
    rng = np.random.default_rng(now.year * 10_000 + now.month * 100 + now.day + now.hour)

    # Training volume and commit count are independent: always train 2-3
    # batches, then emit however many commits were asked for. --commits 1
    # squashes the whole run into a single commit; the default keeps the
    # old one-commit-per-stage behavior.
    if args.commits is None:
        n_batches = int(rng.integers(2, 4))
        train_commit_budget = n_batches
        split_eval = True
    else:
        n_batches = int(rng.integers(2, 4))
        train_commit_budget = max(0, args.commits - 2)
        split_eval = args.commits >= 2
    print(f"run at {now.isoformat()} -> {n_batches} batch(es), "
          f"commits target {args.commits or 'per-stage'}")

    pending = []  # accumulated stage summaries awaiting a commit

    def flush(subject=None):
        if not pending:
            return
        msg = subject or pending[0]
        if len(pending) > 1:
            msg += "\n\n" + "\n".join(f"- {p}" for p in pending)
        git_commit(msg, args.no_commit)
        pending.clear()

    last_train, last_summary, holdout = None, None, None
    total_games = total_positions = 0
    for b in range(n_batches):
        X, policy, value, summary = gather_batch(n_players=2)
        if X is None:
            print(f"batch {b}: no new games ({summary})")
            continue
        # hold out 10% for evaluation (never trained on)
        n_hold = max(1, len(X) // 10)
        holdout = (X[:n_hold], policy[:n_hold])
        stats = train_on_batch(model, opt, meta, X[n_hold:], policy[n_hold:], value[n_hold:])
        save_model(model, opt, meta)
        last_train, last_summary = stats, summary
        total_games += summary["games"]
        total_positions += summary["positions"]
        pending.append(
            f"train: {summary['games']} games / {summary['positions']} positions, "
            f"loss {stats['loss']:.3f} @ step {stats['steps']}")
        if train_commit_budget > 0:
            flush()
            train_commit_budget -= 1

    state_path = Path("data/state.json")
    if last_train is None:
        # No new games this run: the ingest cursor still advanced, so commit
        # that so the working tree is clean for the workflow's push step.
        # Track consecutive dry runs and fail LOUDLY after ~1.5 days of them:
        # a quietly green pipeline that has stopped learning is the worst
        # failure mode (it hid a 15-day ingestion outage in July 2026).
        state = json.loads(state_path.read_text())
        state["dry_runs"] = state.get("dry_runs", 0) + 1
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"no new games this run ({state['dry_runs']} dry runs in a row)")
        git_commit("data: advance ingest cursor (no new games this run)", args.no_commit)
        if state["dry_runs"] >= 8:
            # Flag rather than exit: the workflow fails the run AFTER the
            # push step, so the alarm never blocks commits from landing.
            Path(".starvation_alarm").write_text(
                f"{state['dry_runs']} consecutive runs with no games\n")
        return
    state = json.loads(state_path.read_text())
    if state.get("dry_runs"):
        state["dry_runs"] = 0
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    acc = move_accuracy(model, *holdout)
    row = {
        "timestamp": now.isoformat(timespec="seconds"),
        **{k: last_train[k]
           for k in ("steps", "samples_seen", "loss", "policy_loss", "value_loss")},
        "games": last_summary["games"],
        "positions": last_summary["positions"],
        **{k: round(acc[k], 4) for k in ("top1", "top3")},
    }
    METRICS_DIR.mkdir(exist_ok=True)
    LATEST.write_text(json.dumps({**row, **acc}, indent=2) + "\n")
    pending.append(
        f"eval: top-1 {acc['top1']:.1%} / top-3 {acc['top3']:.1%} "
        f"on {acc['eval_positions']} held-out positions")
    if split_eval:
        flush()

    # tactics puzzle suite (fixed, bucketed by difficulty)
    puz = evaluate_puzzles(model)
    row.update(puz)
    append_history(row)
    LATEST.write_text(json.dumps({**row, **acc}, indent=2) + "\n")
    if puz:
        by_bucket = " ".join(
            f"{k.replace('puzzle_', '')}:{v:.0%}"
            for k, v in puz.items()
            if k.startswith("puzzle_") and k not in ("puzzle_overall", "puzzle_n")
        )
        pending.append(
            f"puzzles: {puz['puzzle_overall']:.1%} overall on {puz['puzzle_n']} "
            f"tactics  [{by_bucket}]")
    # final flush: everything not yet committed lands here. With --commits 1
    # this is the whole run in one commit.
    if len(pending) > 1:
        flush(f"training update: {total_games} games / {total_positions} positions, "
              f"top-1 {acc['top1']:.1%}, puzzles {puz.get('puzzle_overall', 0):.1%}")
    else:
        flush()

    if args.chart:
        make_chart()
        git_commit("chart: refresh training curves", args.no_commit)


if __name__ == "__main__":
    main()
