#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ArmaTerimi Stage-2 champion image - jury-faithful end-to-end test.
# NEEDS THE GPU. Do not start it while training is running (it refuses unless
# the GPU is idle or FORCE=1 is set).
#
# It invokes the FROZEN run_solution.sh, so the docker run flags are the
# jury's by construction (--rm --gpus all --network none --read-only
# --user $(id -u):$(id -g) --cap-drop ALL --security-opt
# no-new-privileges:true --memory 70g --memory-swap 70g --tmpfs /tmp:...8g,
# single-file input bind, /output dir bind, ZERO env vars) - and the timing
# matches the official measurement window: script launch -> output complete.
#
# Usage:
#   bash test_like_jury.sh                 # public_test.csv -> jury_out/submission.csv
#   INPUT=/path/x.csv OUT=/path/sub.csv bash test_like_jury.sh
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/jupyter/trendyol-teknofest-work/stage2_data"
INPUT="${INPUT:-${DATA_DIR}/public_test.csv}"
OUT="${OUT:-${SCRIPT_DIR}/jury_out/submission.csv}"
export CONTAINER_NAME="${CONTAINER_NAME:-armaterimi_solution}"

[[ -f "${INPUT}" ]] || { printf 'input CSV not found: %s\n' "${INPUT}" >&2; exit 2; }
docker image inspect "${CONTAINER_NAME}" >/dev/null 2>&1 \
  || { printf 'image %s not built - run build_docker.sh first\n' "${CONTAINER_NAME}" >&2; exit 2; }

# --- GPU must be idle: this box shares one A100 with training ------------------
if [[ -z "${FORCE:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${busy}" ]]; then
    printf 'REFUSING: GPU has active compute processes (training?):\n%s\nSet FORCE=1 to override.\n' "${busy}" >&2
    exit 3
  fi
fi

OUT_DIR="$(dirname -- "${OUT}")"
mkdir -p "${OUT_DIR}"
rm -f "${OUT}" "${OUT_DIR}"/.*.invalid "${OUT}".tmp* 2>/dev/null || true

# --- the jury's exact command, timed over the official window ------------------
printf '[jury-test] image: %s\n[jury-test] input: %s\n[jury-test] output: %s\n' \
  "${CONTAINER_NAME}" "${INPUT}" "${OUT}"
T0=$(date +%s)
RC=0
bash "${SCRIPT_DIR}/run_solution.sh" "${INPUT}" "${OUT}" || RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))
printf '[jury-test] run_solution.sh exit=%s wall=%ss\n' "${RC}" "${WALL}"
[[ "${RC}" -eq 0 ]] || exit "${RC}"

# --- validate the submission exactly the way evaluate.py will ------------------
python3 - "${INPUT}" "${OUT}" <<'PY'
import csv, hashlib, pathlib, sys
inp, out = sys.argv[1], sys.argv[2]
raw = pathlib.Path(out).read_bytes()
assert not raw.startswith(b"\xef\xbb\xbf"), "BOM present"
with open(inp, newline="", encoding="utf-8") as f:
    rd = csv.reader(f); hdr = next(rd); i = hdr.index("row_id")
    in_ids = [r[i] for r in rd]
with open(out, newline="", encoding="utf-8") as f:
    rd = csv.reader(f); hdr = next(rd)
    assert hdr == ["row_id", "real_label"], hdr
    rows = list(rd)
assert len(rows) == len(in_ids), (len(rows), len(in_ids))
if "public_test" in inp:
    assert len(rows) == 7500, len(rows)
out_ids = [r[0] for r in rows]
assert len(set(out_ids)) == len(out_ids), "duplicate row_id"
assert set(out_ids) == set(in_ids), "row_id set mismatch"
allowed = {"irrelevant", "semi-relevant", "relevant"}
labels = [r[1] for r in rows]
assert set(labels) <= allowed, sorted(set(labels) - allowed)
dist = {k: labels.count(k) for k in sorted(allowed)}
print(f"[jury-test] schema OK: {len(rows)} rows, labels {dist}")
print(f"[jury-test] sha256 {hashlib.sha256(raw).hexdigest()}")
PY

stat -c '[jury-test] output file: owner=%U:%G (%u:%g) mode=%a size=%s bytes' "${OUT}"

# --- nothing but the submission may be left behind -----------------------------
leftovers="$(find "${OUT_DIR}" -maxdepth 1 -type f ! -path "${OUT}" 2>/dev/null || true)"
[[ -z "${leftovers}" ]] || printf '[jury-test] WARNING: leftovers in output dir:\n%s\n' "${leftovers}"

# --- official score, when the public solutions are available -------------------
if [[ -f "${DATA_DIR}/evaluate.py" && -f "${DATA_DIR}/public_solution.csv" ]]; then
  python3 "${DATA_DIR}/evaluate.py" "${OUT}" --solution "${DATA_DIR}/public_solution.csv" || true
fi

ROWS=$(( $(wc -l < "${OUT}") - 1 ))
printf '[jury-test] PASS: %s rows in %ss' "${ROWS}" "${WALL}"
if [[ "${ROWS}" -gt 0 && "${WALL}" -gt 0 ]]; then
  printf '  (~%s rows/s; naive 90k extrapolation ~%ss vs the 9000s hard gate)' \
    "$(( ROWS / WALL ))" "$(( WALL * 90000 / ROWS ))"
fi
printf '\n[jury-test] NOTE: the first run of the day (cold page cache) is the honest one; run twice.\n'
