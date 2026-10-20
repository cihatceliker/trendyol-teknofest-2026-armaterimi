#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ArmaTerimi Stage-2 champion image - container entrypoint.
# argv contract (set by the two FROZEN organizer scripts, which we never edit):
#   solution      : /input/input.csv /output/<name>
#   explainability: serve-ngrok /input/public_test.csv
# Runtime: --read-only rootfs, arbitrary uid:gid with NO passwd entry, no env
# injected except NGROK_AUTHTOKEN (explainability only). Writable: /tmp (8 GB
# tmpfs, EMPTY at start) and /output (solution mode only).
# ---------------------------------------------------------------------------
set -Eeuo pipefail
umask 022                     # the submission CSV must land world-readable
# NEVER `set -x` here: NGROK_AUTHTOKEN may be in this process's environment.

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
# sub-second stamps: the scored window has ~8.6s OUTSIDE the server's own clock and we need to
# know how much is docker/NVIDIA-hook start vs entrypoint vs python boot.
ENTRY_T0=$(date +%s.%N); printf '[entrypoint-stamp] entry_start=%s\n' "$ENTRY_T0" 
die() { printf '[FATAL] %s\n' "$1" >&2; exit "${2:-1}"; }

# --- 1. writable scratch root (do not assume the wrapper mounted the tmpfs) ---
SCRATCH=""
for cand in "${TMPDIR:-/tmp}" /tmp /var/tmp; do
  if mkdir -p "$cand/.probe.$$" 2>/dev/null; then rmdir "$cand/.probe.$$"; SCRATCH="$cand"; break; fi
done
[ -n "$SCRATCH" ] || die "no writable scratch directory (read-only rootfs and /tmp not writable)" 1

CACHE_VARS="HOME TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME
            XDG_RUNTIME_DIR HF_HOME HF_HUB_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE
            TRANSFORMERS_CACHE TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
            TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH MPLCONFIGDIR"
if [ "$SCRATCH" != "/tmp" ]; then
  log "WARNING: scratch root is $SCRATCH (not /tmp); relocating /tmp-pointing cache vars"
  for v in $CACHE_VARS; do
    cur="${!v-}"
    case "$cur" in /tmp/*|/tmp) export "$v=${SCRATCH}${cur#/tmp}" ;; esac
  done
fi

# --- 2. materialise every cache dir, $HOME included (tmpfs is EMPTY at start) -
for v in $CACHE_VARS; do d="${!v-}"; [ -n "$d" ] && mkdir -p "$d"; done
chmod 0700 "${XDG_RUNTIME_DIR:-$SCRATCH/run}" 2>/dev/null || true

log "uid=$(id -u) gid=$(id -g) HOME=${HOME:-unset} scratch=$SCRATCH mode=${1:-solution}"

mode="${1:-}"

# --- 3. explainability dispatch: argv[1] == "serve-ngrok" (or local "serve") --
# Rules 6.4: same model + preprocessing as scoring, local UI + in-container
# ngrok, fail closed without NGROK_AUTHTOKEN, print the public URL, never the
# token, stop both on exit. Startup order (10-minute link budget, contract
# 3.3): UI binds and answers /healthz BEFORE the model loads (background
# thread inside app.py), tunnel comes up right after -> the link is live in
# well under a minute; the model finishes loading behind it.
if [ "$mode" = "serve-ngrok" ] || [ "$mode" = "serve" ]; then
  shift
  csv="${1:-}"; [ -n "$csv" ] || die "public test CSV path is required" 2
  [ -r "$csv" ] || die "cannot read public test CSV: $csv" 2

  if [ "$mode" = "serve-ngrok" ]; then
    # RULES 6.4: fail closed, before loading anything. Exit 2 matches the
    # frozen script's own usage code. Never print or persist the token.
    [ -n "${NGROK_AUTHTOKEN:-}" ] \
      || die "NGROK_AUTHTOKEN is required for explainability mode (fail-closed)" 2
  fi

  # device: baked EXPLAIN_DEVICE=auto -> cuda when the judge ran --gpus all,
  # cpu otherwise (the app is fully functional on CPU; explain ~2-4 s/click).
  dev="${EXPLAIN_DEVICE:-auto}"
  if [ "$dev" = "auto" ]; then
    if python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null
    then dev="cuda"; else dev="cpu"; fi
  fi
  port="${APP_PORT:-7860}"
  model_dir="${MODEL_DIR:-/models/current}"
  log "explainability: device=$dev port=$port model=$model_dir csv=$csv"

  # helper scripts land on the tmpfs (rootfs is read-only; python-only, no curl)
  HLP="$SCRATCH/exp"
  mkdir -p "$HLP" "$SCRATCH/ngrok"

  cat > "$HLP/wait_http.py" <<'PY'
import sys, time, urllib.request
url, deadline = sys.argv[1], time.time() + float(sys.argv[2])
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        pass
    time.sleep(0.5)
sys.exit(1)
PY

  cat > "$HLP/tunnel_url.py" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
        d = json.load(r)
    for t in d.get("tunnels", []):
        u = t.get("public_url", "")
        if u.startswith("https://"):
            print(u)
            break
except Exception:
    pass
PY

  cat > "$HLP/model_state.py" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as r:
        print(json.load(r).get("model", "?"))
except Exception:
    print("?")
PY

  # tail of the ngrok agent log with the token redacted, for diagnostics only
  # (the token is read from the ENVIRONMENT inside python - never argv/files)
  cat > "$HLP/ngrok_diag.py" <<'PY'
import os, sys
tok = os.environ.get("NGROK_AUTHTOKEN", "")
try:
    lines = open(sys.argv[1], errors="replace").readlines()[-15:]
except OSError:
    lines = ["<no ngrok agent log>\n"]
for l in lines:
    sys.stderr.write(l.replace(tok, "[REDACTED]") if tok else l)
PY

  app_args=( "$csv" --model-dir "$model_dir" --device "$dev"
             --max-len "${MAXLEN:-256}" --host 127.0.0.1 --port "$port" )

  if [ "$mode" = "serve" ]; then
    # local test mode (no tunnel): bind on all interfaces so `docker run -p` works
    exec python /app/explain/scripts/app.py "${app_args[@]/127.0.0.1/0.0.0.0}"
  fi

  # ---- supervised UI + tunnel startup (serve-ngrok) --------------------------
  app_pid=""; ngrok_pid=""
  cleanup() {
    trap - EXIT INT TERM
    for p in $app_pid $ngrok_pid; do kill -TERM "$p" 2>/dev/null || true; done
    for _ in 1 2 3; do
      alive=0
      for p in $app_pid $ngrok_pid; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && break
      sleep 1
    done
    for p in $app_pid $ngrok_pid; do kill -KILL "$p" 2>/dev/null || true; done
    for p in $app_pid $ngrok_pid; do wait "$p" 2>/dev/null || true; done
    log "UI and tunnel stopped"
  }
  trap cleanup EXIT INT TERM

  # 1) the UI (same checkpoint + preprocessing as scoring; binds before model load)
  python /app/explain/scripts/app.py "${app_args[@]}" --require-token &
  app_pid=$!

  python "$HLP/wait_http.py" "http://127.0.0.1:$port/healthz" 120 \
    || die "UI failed to answer /healthz within 120s" 1
  log "UI is up on 127.0.0.1:$port (model loading in background)"

  # 2) the tunnel. ngrok v3 reads NGROK_AUTHTOKEN from the ENVIRONMENT (never
  #    argv - /proc/*/cmdline is world-readable; never a config file). Raw
  #    agent output goes to the tmpfs log, never to stdout (rules 6.4).
  url=""
  for attempt in 1 2 3; do
    /app/vendor/ngrok http "127.0.0.1:$port" \
      --log "$SCRATCH/ngrok/agent.log" --log-format logfmt --log-level info \
      >/dev/null 2>&1 &
    ngrok_pid=$!
    for _ in $(seq 1 45); do
      kill -0 "$ngrok_pid" 2>/dev/null || break
      url="$(python "$HLP/tunnel_url.py")"
      [ -n "$url" ] && break
      sleep 2
    done
    [ -n "$url" ] && break
    log "tunnel attempt $attempt failed; ngrok agent log tail (redacted):"
    python "$HLP/ngrok_diag.py" "$SCRATCH/ngrok/agent.log" || true
    kill -KILL "$ngrok_pid" 2>/dev/null || true
    wait "$ngrok_pid" 2>/dev/null || true
    ngrok_pid=""
    sleep 3
  done
  [ -n "$url" ] || die "could not establish the ngrok tunnel after 3 attempts (bad token or blocked egress?)" 1

  printf '\n================================================================\n'
  printf '  EXPLAINABILITY UI (public ngrok URL):\n'
  printf '\n      %s\n\n' "$url"
  printf '  health: %s/healthz   ready: %s/readyz\n' "$url" "$url"
  printf '================================================================\n\n'
  log "NGROK_URL=$url"

  # 3) report model readiness (bounded; the link is already live - R8 is met)
  for _ in $(seq 1 60); do
    kill -0 "$app_pid" 2>/dev/null || break
    st="$(python "$HLP/model_state.py" "$port")"
    if [ "$st" = "ready" ]; then log "model is READY - Explain is live"; break; fi
    if [ "$st" = "error" ]; then
      die "model failed to load inside the UI (see app log above)" 1
    fi
    sleep 5
  done

  # 4) supervise: the death of either process takes the container down (a live
  #    tunnel in front of a dead UI would show the judge a silent 502).
  set +e
  wait -n "$app_pid" "$ngrok_pid"
  rc=$?
  set -e
  if kill -0 "$app_pid" 2>/dev/null; then
    log "ngrok agent exited (rc=$rc); shutting down"
  else
    log "UI process exited (rc=$rc); shutting down"
  fi
  exit "$rc"
fi

# --- 4. Solution mode: exactly two positional arguments -----------------------
[ "$#" -eq 2 ] || die "usage: <INPUT_CSV_PATH> <OUTPUT_CSV_PATH> (got $# argument(s))" 2
in_csv="$1"; out_csv="$2"
[ -r "$in_csv" ] || die "cannot read input CSV: $in_csv" 2
out_dir="$(dirname -- "$out_csv")"
{ [ -d "$out_dir" ] && [ -w "$out_dir" ]; } || die "output directory is not writable: $out_dir" 2

# exec => Python becomes PID 1: receives SIGTERM/SIGINT directly, no zombie
# reaping needed. The output filename comes from argv and is NEVER hardcoded.
printf '[entrypoint-stamp] entry_wall_before_python=%s\n' "$(awk -v a="$ENTRY_T0" 'BEGIN{printf "%.3f", systime()-a}')"
printf '[entrypoint-stamp] exec_python=%s\n' "$(date +%s.%N)"
exec python /app/src/predict.py "$in_csv" "$out_csv"
