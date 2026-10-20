#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ArmaTerimi Stage-2 champion image - stage the build context from the
# READ-ONLY experiment directories. Run before build_docker.sh, and again
# whenever the champion checkpoint changes.
#
# The checkpoint is PARAMETERIZED: point MODEL_SRC at any serve_hf-compatible
# checkpoint dir (model.safetensors + config.json + tokenizer.json +
# modeling.py + configuration.py [+ decision.json]) and re-run; nothing else
# in the scaffold ever changes. In-image path stays /models/current.
#
# Usage:
#   bash prepare_context.sh                                  # default champion
#   MODEL_SRC=/path/to/other_ckpt bash prepare_context.sh    # checkpoint swap
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SRC="${MODEL_SRC:-/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/outputs_frozen_v2/ckpt_r1_oldstore}"
SERVE_SRC="${SERVE_SRC:-/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e12_lean_infer/scripts}"
EXPLAIN_SRC="${EXPLAIN_SRC:-/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e20_explain_ty}"
COMMON_SRC="${COMMON_SRC:-/home/jupyter/trendyol-teknofest-work/asya/s2_exp/common}"
E18_SRC="${E18_SRC:-/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e18_attr_clean}"

# --- vendored ngrok v3 agent (explainability tunnel; runtime downloads are
#     forbidden). PINNED: version 3.39.11, linux-amd64, static Go binary.
#     Provenance (2026-08-24): stable tgz sha256 cec0b499... from
#     bin.equinox.io; the SAME binary (sha256 below) is inside
#     ngrok_3.39.11-0_amd64.deb, whose sha256 e51aa234... is PUBLISHED in
#     ngrok's own apt-repo Packages index (ngrok-agent.s3.amazonaws.com).
#     A version bump is a DELIBERATE act: update all three pins together.
NGROK_VERSION="3.39.11"
NGROK_TGZ_URL="${NGROK_TGZ_URL:-https://bin.equinox.io/a/5Mrhp7gRDVb/ngrok-v3-3.39.11-linux-amd64.tar.gz}"
NGROK_TGZ_SHA256="cec0b4997fcc5f529dfc74bac89050354d11a915f968720600039738fdf330cf"
NGROK_BIN_SHA256="e4e0f05c3b016699daa3fc4ff03d0870e49f705171d2dac5f16d00c55c10dee5"

# --- 0. sources must exist and be complete ------------------------------------
# modeling.py/configuration.py exist ONLY for remote-code checkpoints (the GTE "New" family,
# i.e. TY-ecomm). Plain BERT checkpoints (BERTurk, the turkish-* tinies) have neither and are
# served by lean_infer.py, which reads safetensors directly. Requiring them unconditionally made
# prepare_context reject every non-TY candidate.
REMOTE_CODE=0
grep -q '"auto_map"' "${MODEL_SRC}/config.json" 2>/dev/null && REMOTE_CODE=1
need=(model.safetensors config.json tokenizer.json)
[[ "${REMOTE_CODE}" -eq 1 ]] && need+=(modeling.py configuration.py)
for f in "${need[@]}"; do
  [[ -s "${MODEL_SRC}/${f}" ]] \
    || { printf 'MODEL_SRC is not a servable checkpoint: missing %s/%s\n' "${MODEL_SRC}" "${f}" >&2; exit 2; }
done
printf '[prepare] %s: remote_code=%s\n' "$(basename "${MODEL_SRC}")" "${REMOTE_CODE}"
[[ -s "${MODEL_SRC}/decision.json" ]] \
  || printf 'WARNING: %s has no decision.json - lean_infer.decide() will use its softmax3/id2label fallback\n' "${MODEL_SRC}" >&2
for f in serve_hf.py lean_infer.py serve_gte.py lean_gte.py; do
  [[ -s "${SERVE_SRC}/${f}" ]] \
    || { printf 'SERVE_SRC is missing %s/%s\n' "${SERVE_SRC}" "${f}" >&2; exit 2; }
done
for f in scripts/app.py scripts/model_adapter.py static/index.html; do
  [[ -s "${EXPLAIN_SRC}/${f}" ]] \
    || { printf 'EXPLAIN_SRC is missing %s/%s\n' "${EXPLAIN_SRC}" "${f}" >&2; exit 2; }
done
for f in normalize.py drop_keys.csv; do
  [[ -s "${COMMON_SRC}/${f}" ]] \
    || { printf 'COMMON_SRC is missing %s/%s\n' "${COMMON_SRC}" "${f}" >&2; exit 2; }
done
for f in attr_clean.py drop_keys_v2.csv; do
  [[ -s "${E18_SRC}/${f}" ]] \
    || { printf 'E18_SRC is missing %s/%s\n' "${E18_SRC}" "${f}" >&2; exit 2; }
done

# --- 1. refuse to stage anything that must never enter an image ---------------
# (data files, credential files; the image carries model assets + code ONLY)
banned="$(find -L "${MODEL_SRC}" -maxdepth 1 -type f \
  \( -name '*.parquet' -o -name '*.csv' -o -iname '*openrouter*' -o -iname '*ngrok*' \
     -o -name '*.pem' -o -name '.netrc' -o -name 'token' -o -name '.env*' \) 2>/dev/null || true)"
if [[ -n "${banned}" ]]; then
  printf 'REFUSING: MODEL_SRC contains files that must not enter the image:\n%s\n' "${banned}" >&2
  exit 2
fi

# --- 2. model -> models/current/ (symlinks resolved; --delete => clean swap) --
mkdir -p "${SCRIPT_DIR}/models/current"
printf '[prepare] staging checkpoint %s -> models/current/ ...\n' "${MODEL_SRC}"
rsync -aL --delete "${MODEL_SRC}/" "${SCRIPT_DIR}/models/current/"

# --- 3. serving pair -> src/ (byte-identical copies; originals stay untouched) -
cp -f "${SERVE_SRC}/serve_hf.py" "${SERVE_SRC}/lean_infer.py" \
      "${SERVE_SRC}/serve_gte.py" "${SERVE_SRC}/lean_gte.py" "${SCRIPT_DIR}/src/"

# --- 3b. explainability app + preprocessing sources -> explain/ ---------------
# Layout mirrors what the baked E20_COMMON_DIR/E20_E18_DIR expect AND keeps
# attr_clean.py's own "../common" sys.path fallback valid:
#   explain/scripts/{app.py,model_adapter.py}   explain/static/index.html
#   explain/common/{normalize.py,drop_keys.csv} explain/e18_attr_clean/{attr_clean.py,drop_keys_v2.csv}
mkdir -p "${SCRIPT_DIR}/explain/scripts" "${SCRIPT_DIR}/explain/static" \
         "${SCRIPT_DIR}/explain/common" "${SCRIPT_DIR}/explain/e18_attr_clean"
cp -f "${EXPLAIN_SRC}/scripts/app.py" "${EXPLAIN_SRC}/scripts/model_adapter.py" "${SCRIPT_DIR}/explain/scripts/"
cp -f "${EXPLAIN_SRC}/static/index.html" "${SCRIPT_DIR}/explain/static/"
cp -f "${COMMON_SRC}/normalize.py" "${COMMON_SRC}/drop_keys.csv" "${SCRIPT_DIR}/explain/common/"
cp -f "${E18_SRC}/attr_clean.py" "${E18_SRC}/drop_keys_v2.csv" "${SCRIPT_DIR}/explain/e18_attr_clean/"

# --- 3c. vendored ngrok agent -> vendor/ngrok (download only when absent/stale)
mkdir -p "${SCRIPT_DIR}/vendor"
ngrok_ok=0
if [[ -s "${SCRIPT_DIR}/vendor/ngrok" ]] \
   && printf '%s  %s\n' "${NGROK_BIN_SHA256}" "${SCRIPT_DIR}/vendor/ngrok" | sha256sum -c --quiet - 2>/dev/null; then
  ngrok_ok=1
fi
if [[ "${ngrok_ok}" -ne 1 ]]; then
  printf '[prepare] vendoring ngrok v%s (linux-amd64) from %s ...\n' "${NGROK_VERSION}" "${NGROK_TGZ_URL}"
  tmp_tgz="${SCRIPT_DIR}/vendor/.ngrok.tgz.part"
  curl -fsSL --max-time 300 -o "${tmp_tgz}" "${NGROK_TGZ_URL}"
  printf '%s  %s\n' "${NGROK_TGZ_SHA256}" "${tmp_tgz}" | sha256sum -c --quiet - \
    || { printf 'ngrok tarball sha256 MISMATCH (a new release replaced the pinned one?) - refusing.\n' >&2; rm -f "${tmp_tgz}"; exit 2; }
  tar -xzf "${tmp_tgz}" -C "${SCRIPT_DIR}/vendor" ngrok
  rm -f "${tmp_tgz}"
fi
printf '%s  %s\n' "${NGROK_BIN_SHA256}" "${SCRIPT_DIR}/vendor/ngrok" | sha256sum -c --quiet - \
  || { printf 'vendored ngrok binary sha256 MISMATCH - refusing.\n' >&2; exit 2; }
printf '%s  ngrok\n' "${NGROK_BIN_SHA256}" > "${SCRIPT_DIR}/vendor/ngrok.sha256"
chmod 0755 "${SCRIPT_DIR}/vendor/ngrok"

# --- 4. secrets scan over the whole context (fail early, before any build) ----
leak="$(LC_ALL=C grep -rIlE 'sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|BEGIN (RSA|EC|OPENSSH|PRIVATE)' \
        "${SCRIPT_DIR}" 2>/dev/null || true)"
if [[ -n "${leak}" ]]; then
  printf 'REFUSING: secret-shaped content found in the context:\n%s\n' "${leak}" >&2
  exit 2
fi

# --- 5. provenance manifest (README / final report; NOT copied into the image) -
{
  printf 'staged_at_utc: %s\n' "$(date -u +%FT%TZ)"
  printf 'model_src: %s\n' "${MODEL_SRC}"
  printf 'serve_src: %s\n' "${SERVE_SRC}"
  printf 'model_file_sha256:\n'
  ( cd "${SCRIPT_DIR}/models/current" && sha256sum -- * | sed 's/^/  /' )
  printf 'src_file_sha256:\n'
  ( cd "${SCRIPT_DIR}/src" && sha256sum -- *.py | sed 's/^/  /' )
  printf 'entrypoint_sha256:\n'
  ( cd "${SCRIPT_DIR}" && sha256sum entrypoint.sh | sed 's/^/  /' )
  printf 'explain_src: %s\n' "${EXPLAIN_SRC}"
  printf 'explain_file_sha256:\n'
  ( cd "${SCRIPT_DIR}" && sha256sum explain/scripts/*.py explain/static/index.html \
      explain/common/normalize.py explain/common/drop_keys.csv \
      explain/e18_attr_clean/attr_clean.py explain/e18_attr_clean/drop_keys_v2.csv | sed 's/^/  /' )
  printf 'ngrok_version: %s\n' "${NGROK_VERSION}"
  printf 'ngrok_url: %s\n' "${NGROK_TGZ_URL}"
  printf 'ngrok_sha256:\n'
  ( cd "${SCRIPT_DIR}" && sha256sum vendor/ngrok | sed 's/^/  /' )
} > "${SCRIPT_DIR}/models/STAGING_MANIFEST.txt"

# --- 6. world-readable, world-traversable - on the HOST, never via RUN chown
#        (a RUN chown/chmod -R over /models would duplicate the layer) ---------
chmod -R a+rX "${SCRIPT_DIR}"
chmod 0755 "${SCRIPT_DIR}"/*.sh

printf '[prepare] done. Context size:\n'
du -sh "${SCRIPT_DIR}/models" "${SCRIPT_DIR}/src" "${SCRIPT_DIR}/explain" "${SCRIPT_DIR}/vendor"
printf '[prepare] next: bash build_docker.sh\n'
