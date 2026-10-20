#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ArmaTerimi Stage-2 champion image - OUR build script. The organizer
# template's global prune block is deliberately absent: rules 6.2 reject host
# administration / global Docker cleanup even when it appears in a reference
# implementation. This script does ONLY: validate the context, scan for
# secrets, docker build, tag, audit, report.
#
# Usage:  [CONTAINER_NAME=armaterimi_solution] bash build_docker.sh
# ---------------------------------------------------------------------------
set -Eeuo pipefail

# Docker image names must be lowercase (team ArmaTerimi -> armaterimi_solution).
CONTAINER_NAME="${CONTAINER_NAME:-armaterimi_solution}"
if [[ "${CONTAINER_NAME}" =~ [A-Z] ]]; then
  printf 'image name must be lowercase (got %s); use e.g. armaterimi_solution\n' "${CONTAINER_NAME}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is not installed or not on PATH\n' >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  printf 'cannot query the Docker daemon; verify Docker is running and accessible\n' >&2
  exit 2
fi

fail=0

# --- 1. frozen organizer scripts must be byte-identical (contract 1.1) --------
expect_sha() { # file expected_sha
  local got
  got="$(sha256sum "$1" | awk '{print $1}')"
  if [[ "${got}" != "$2" ]]; then
    printf 'FROZEN SCRIPT MODIFIED: %s\n  expected %s\n  got      %s\n' "$1" "$2" "${got}" >&2
    fail=1
  fi
}
expect_sha "${SCRIPT_DIR}/run_solution.sh"       "8dbe185ccf028905f31ce860100e8fc139464990d97a37013f10d8ffad2b03e3"
expect_sha "${SCRIPT_DIR}/run_explainability.sh" "ec170097893e852e6ab9b8a71721a1fcbae0a156fa38a0260395a4af24b7250a"

# --- 2. context must be staged (prepare_context.sh ran) ------------------------
# modeling.py/configuration.py are staged ONLY for remote-code checkpoints (GTE "New" family).
# A plain-BERT checkpoint has neither and is served by lean_infer.py; requiring them here made
# every non-TY candidate unbuildable.
REMOTE_CODE_FILES=()
grep -q '"auto_map"' "${SCRIPT_DIR}/models/current/config.json" 2>/dev/null \
  && REMOTE_CODE_FILES=("${SCRIPT_DIR}/models/current/modeling.py" "${SCRIPT_DIR}/models/current/configuration.py")
for path in \
  "${SCRIPT_DIR}/models/current/model.safetensors" \
  "${SCRIPT_DIR}/models/current/config.json" \
  "${SCRIPT_DIR}/models/current/tokenizer.json" \
  "${REMOTE_CODE_FILES[@]}" \
  "${SCRIPT_DIR}/src/serve_hf.py" \
  "${SCRIPT_DIR}/src/lean_infer.py" \
  "${SCRIPT_DIR}/src/serve_gte.py" \
  "${SCRIPT_DIR}/src/lean_gte.py" \
  "${SCRIPT_DIR}/src/predict.py" \
  "${SCRIPT_DIR}/explain/scripts/app.py" \
  "${SCRIPT_DIR}/explain/scripts/model_adapter.py" \
  "${SCRIPT_DIR}/explain/static/index.html" \
  "${SCRIPT_DIR}/explain/common/normalize.py" \
  "${SCRIPT_DIR}/explain/common/drop_keys.csv" \
  "${SCRIPT_DIR}/explain/e18_attr_clean/attr_clean.py" \
  "${SCRIPT_DIR}/explain/e18_attr_clean/drop_keys_v2.csv" \
  "${SCRIPT_DIR}/vendor/ngrok" \
  "${SCRIPT_DIR}/vendor/ngrok.sha256" \
  "${SCRIPT_DIR}/entrypoint.sh" \
  "${SCRIPT_DIR}/requirements.txt"; do
  if [[ ! -s "${path}" ]]; then
    printf 'build context is not staged: missing %s\n  -> run: bash %s/prepare_context.sh\n' \
      "${path}" "${SCRIPT_DIR}" >&2
    fail=1
  fi
done

# --- 3. secrets scan (refuse to build on any hit) ------------------------------
#   a) key-shaped content anywhere in the context (text files; binaries are
#      staged only from the checkpoint dir and covered by the name checks)
leak="$(LC_ALL=C grep -rIlE 'sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|BEGIN (RSA|EC|OPENSSH|PRIVATE)' \
        "${SCRIPT_DIR}" 2>/dev/null || true)"
if [[ -n "${leak}" ]]; then
  printf 'SECRETS CHECK FAILED - key-shaped content in:\n%s\n' "${leak}" >&2
  fail=1
fi
#   b) credential-carrying filenames (openrouter.txt / ngrok.txt live in
#      /home/jupyter and must NEVER be copied here)
badnames="$(find "${SCRIPT_DIR}" -type f ! -path "${SCRIPT_DIR}/vendor/*" \
            \( -iname '*openrouter*' -o -iname '*ngrok*' \
            -o -name '*.pem' -o -name '.netrc' -o -name 'token' -o -name '.env*' \) 2>/dev/null || true)"
if [[ -n "${badnames}" ]]; then
  printf 'SECRETS CHECK FAILED - forbidden filenames in the context:\n%s\n' "${badnames}" >&2
  fail=1
fi
#   b2) vendor/ is exempt from the name check above ONLY because it must hold
#       exactly the sha-pinned ngrok agent binary + its sha file, nothing else
#       (in particular no *.txt, no config, no token-shaped anything).
NGROK_BIN_SHA256="e4e0f05c3b016699daa3fc4ff03d0870e49f705171d2dac5f16d00c55c10dee5"
vendor_extra="$(find "${SCRIPT_DIR}/vendor" -type f ! -name 'ngrok' ! -name 'ngrok.sha256' 2>/dev/null || true)"
if [[ -n "${vendor_extra}" ]]; then
  printf 'VENDOR CHECK FAILED - unexpected files in vendor/ (only ngrok + ngrok.sha256 allowed):\n%s\n' "${vendor_extra}" >&2
  fail=1
fi
if [[ -s "${SCRIPT_DIR}/vendor/ngrok" ]] \
   && ! printf '%s  %s\n' "${NGROK_BIN_SHA256}" "${SCRIPT_DIR}/vendor/ngrok" | sha256sum -c --quiet - 2>/dev/null; then
  printf 'VENDOR CHECK FAILED - vendor/ngrok does not match the pinned sha256 (re-run prepare_context.sh)\n' >&2
  fail=1
fi
#   b3) the LOCAL runtime ngrok token (never ours to bake): if the credential
#       file exists on this machine, its content must appear NOWHERE in the
#       context. Pattern is passed via process substitution - never argv,
#       never echoed, never written.
if [[ -r /home/jupyter/ngrok.txt ]] \
   && [[ "$(tr -d '[:space:]' < /home/jupyter/ngrok.txt | wc -c)" -ge 20 ]]; then
  if grep -rqF -f <(tr -d '[:space:]' < /home/jupyter/ngrok.txt) "${SCRIPT_DIR}" 2>/dev/null; then
    printf 'SECRETS CHECK FAILED - the local ngrok token content appears inside the build context\n' >&2
    fail=1
  fi
fi
#   c) no data files may enter the image (the .dockerignore also blocks them,
#      but a staged copy would still bloat the submission ZIP)
datafiles="$(find "${SCRIPT_DIR}/models" "${SCRIPT_DIR}/src" \
             -type f \( -name '*.parquet' -o -name '*.csv' \) 2>/dev/null || true)"
if [[ -n "${datafiles}" ]]; then
  printf 'DATA CHECK FAILED - parquet/csv files staged into the context:\n%s\n' "${datafiles}" >&2
  fail=1
fi

# --- 4. world-readable context: the runtime uid is arbitrary, no groups --------
unreadable="$(find "${SCRIPT_DIR}" \( -type f ! -perm -o=r \) -o \( -type d ! -perm -o=x \) 2>/dev/null | head -5 || true)"
if [[ -n "${unreadable}" ]]; then
  printf 'context contains non-world-readable entries (run prepare_context.sh / chmod -R a+rX):\n%s\n' "${unreadable}" >&2
  fail=1
fi

[[ "${fail}" -eq 0 ]] || exit 2

printf '[docker-build] context: %s\n' "${SCRIPT_DIR}"
printf '[docker-build] image:   %s\n' "${CONTAINER_NAME}"

export DOCKER_BUILDKIT=1
BUILD_T0=$SECONDS
docker build \
  --progress=plain \
  --tag "${CONTAINER_NAME}" \
  "${SCRIPT_DIR}"
BUILD_WALL=$((SECONDS - BUILD_T0))

# Belt and braces: run_solution.sh defaults to CONTAINER_NAME=example_submission;
# a judge who forgets to export it still gets a run instead of exit 125.
docker tag "${CONTAINER_NAME}" example_submission

# --- 5. post-build audit: nothing secret-shaped in history or baked env --------
if docker history --no-trunc --format '{{.CreatedBy}}' "${CONTAINER_NAME}" \
     | grep -qiE 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|authtoken|password|secret='; then
  printf '[docker-build] AUDIT FAILED: secret-shaped text in docker history\n' >&2
  exit 1
fi
# (value must look like an actual credential: our own HF_HUB_DISABLE_IMPLICIT_TOKEN=1
#  and similar flag vars are fine)
if docker image inspect "${CONTAINER_NAME}" --format '{{json .Config.Env}}' \
     | grep -qiE '(TOKEN|SECRET|PASSWORD|_KEY)=[A-Za-z0-9+/_-]{12,}'; then
  printf '[docker-build] AUDIT FAILED: secret-shaped variable in image ENV\n' >&2
  exit 1
fi
printf '[docker-build] audit OK: history and ENV are clean\n'

printf '[docker-build] built %s (also tagged example_submission) in %ss\n' "${CONTAINER_NAME}" "${BUILD_WALL}"
docker image inspect "${CONTAINER_NAME}" \
  --format '[docker-build] image id={{.Id}} entrypoint={{json .Config.Entrypoint}}'
docker images --format '[docker-build] size: {{.Size}}' "${CONTAINER_NAME}" | head -1
