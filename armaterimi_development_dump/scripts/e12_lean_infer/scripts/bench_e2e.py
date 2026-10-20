"""E0a end-to-end timing harness — GPU. DO NOT run while the GPU is busy with training.

Launches lean_infer.py as a fresh subprocess (cold start counts: the score is 90000/runtime from
container launch to output-file completion) and reports the stage breakdown lean_infer prints:
imports, read+normalise, tokenize, model-ready, first-batch (= cold start), inference, CSV write.
External wall time is measured around the subprocess, so Python startup itself is included.
Projects the 90k-row score by scaling the row-dependent stages (read/tokenize/infer/write)
linearly and keeping startup fixed. Run twice: run 1 = cold page cache, run 2 = warm.

Usage: bench_e2e.py --model-dir <dir> [--input <csv>] [--runs 2] [--no-overlap]
"""
import os, sys, time, subprocess

W = "/home/jupyter/trendyol-teknofest-work"
PY = f"{W}/.venv-gcp/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("BENCH_SCRIPT", "lean_infer.py"))
TARGET_ROWS = 90_000

def main(argv):
    kw = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "") for a in argv if a.startswith("--"))
    if "model-dir" not in kw:
        sys.exit("usage: bench_e2e.py --model-dir <dir> [--input <csv>] [--runs=2] [--no-overlap]")
    inp = kw.get("input") or f"{W}/stage2_data/public_test.csv"
    out = f"{W}/asya/s2_exp/e12_lean_infer/outputs/bench_pred.csv"
    cmd = [PY, SCRIPT, inp, out, f"--model-dir={kw['model-dir']}"]
    if "no-overlap" in kw: cmd.append("--no-overlap")
    for run in range(1, int(kw.get("runs") or 2) + 1):
        if os.path.exists(out): os.remove(out)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        # the clock stops when the output file exists — mirror the competition's definition
        wall = time.perf_counter() - t0
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
        if r.returncode or not os.path.exists(out):
            sys.exit(f"run {run}: lean_infer failed (rc={r.returncode})")
        st = {}
        for line in r.stdout.splitlines():
            if line.startswith("STAGE "):
                st = dict(kv.split("=") for kv in line.split()[1:])
        rows = int(st["rows"]); inside = float(st["total"])
        startup = wall - (float(st["read"]) + float(st["tokenize"]) + float(st["infer"]) + float(st["write"]))
        scaled = startup + (wall - startup) * TARGET_ROWS / rows
        print(f"--- run {run}: wall {wall:.2f}s (interp+imports overhead {wall-inside:.2f}s)  "
              f"cold-start-to-first-batch {st['first_batch']}s  eff SPS {rows/wall:,.0f}")
        print(f"    projected @90k: {scaled:.1f}s wall -> score {TARGET_ROWS/scaled:,.0f}  "
              f"(startup {startup:.2f}s fixed + {(wall-startup)*TARGET_ROWS/rows:.1f}s scaled)")

if __name__ == "__main__":
    main(sys.argv[1:])
