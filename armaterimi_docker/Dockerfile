# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ArmaTerimi Stage-2 SUBMISSION image (e15 student, BERTurk-128k 12L, 184.3M params).
# ONE image, TWO modes (argv appended by the FROZEN organizer scripts):
#   solution mode        : ENTRYPOINT <INPUT_CSV> <OUTPUT_CSV>      (--network none, --read-only)
#   explainability mode  : ENTRYPOINT serve-ngrok <PUBLIC_TEST_CSV> (--network bridge, --read-only)
#     -> e20 explainability app + vendored sha-pinned ngrok v3 agent (phase 7).
# Runtime user is an ARBITRARY uid:gid from the judge (--user "$(id -u):$(id -g)")
# with NO passwd entry. Writable at runtime: /tmp (8 GB tmpfs) and, in solution
# mode only, /output. ZERO env vars are injected in solution mode - everything
# the runtime needs is baked below.
#
# BEFORE BUILDING: bash prepare_context.sh  (stages models/current/ + src/ from
# the read-only experiment dirs and fixes modes host-side; a RUN chown/chmod
# over the model tree would duplicate the layer).
#
# Base: pytorch/pytorch 2.10.0-cuda12.8-cudnn9-runtime, digest-pinned 2026-08-24
# (Docker Hub API). Ships torch 2.10.0+cu128 == the .venv-gcp stack the champion
# was validated with. Driver on the judging machine is 580.126.20 / CUDA 13.0;
# a CUDA 12.8 userspace runs under it via backward compatibility (contract T10).
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

# ---- 1. Environment: EVERYTHING baked, because nothing is injected at runtime -
# NOTE deviations from the E0 scaffold, both deliberate:
#  * no PYTHONPYCACHEPREFIX: setting it makes Python IGNORE the .pyc baked into
#    site-packages/app layers (cache lookup moves to the prefix), re-parsing the
#    whole import chain on every start. PYTHONDONTWRITEBYTECODE=1 alone
#    guarantees no runtime write attempts against the read-only rootfs.
#  * no TOKENIZERS_PARALLELISM: serve_hf.py setdefaults it to "true" on purpose
#    (Rust parallel encode_batch; the serving path forks no processes, so the
#    fork-after-parallelism warning cannot trigger).
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    # --- writable-path redirection (rootfs is read-only; only /tmp is writable;
    #     uid has no passwd entry so $HOME would default to the read-only /) ---
    HOME=/tmp/home TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/cache XDG_CONFIG_HOME=/tmp/config XDG_DATA_HOME=/tmp/data \
    XDG_STATE_HOME=/tmp/state XDG_RUNTIME_DIR=/tmp/run \
    HF_HOME=/tmp/hf HF_HUB_CACHE=/tmp/hf/hub HF_ASSETS_CACHE=/tmp/hf/assets \
    HF_DATASETS_CACHE=/tmp/hf/datasets TRANSFORMERS_CACHE=/tmp/hf/hub \
    TORCH_HOME=/tmp/torch TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor \
    TRITON_CACHE_DIR=/tmp/cache/triton TORCH_EXTENSIONS_DIR=/tmp/cache/torch_ext \
    CUDA_CACHE_PATH=/tmp/cache/nv MPLCONFIGDIR=/tmp/cache/mpl \
    # --- offline + telemetry kill switches (--network none in scoring) ---
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1 \
    DO_NOT_TRACK=1 DISABLE_TELEMETRY=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    # --- perf / determinism (12 logical CPUs, 1x A100-40GB) ---
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_MODULE_LOADING=LAZY \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    # --- required because the base is NOT an nvidia/cuda:* image (contract 1.2):
    #     without these the NVIDIA hook injects no driver under --gpus all ---
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # --- our knobs, all with baked defaults (read by src/predict.py + serve_hf.py) ---
    SERVE_SCRIPT=serve_gte.py \
    MODEL_DIR=/models/current \
    BATCH_ROWS=256 TOKEN_BUDGET=65536 MAXLEN=256 \
    # --- thread budget. MEASURED 2026-08-26 on the 12-core box: each pool worker's Rust
    #     tokenizer opens its OWN rayon pool, so N workers meant 12N threads on 12 cores and the
    #     model-load thread starved (model_ready 5.93s against a 2.0s uncontended cost). Pinning
    #     rayon+OMP to 1 thread is BIT-EXACT (0/90,000 labels change on every candidate) and worth
    #     +5% to +17% depending on the model. WORKER COUNT re-measured IN-CONTAINER 2026-08-27:
    #     the host optimum (8) does not transfer -- inside the jury's flag set `nproc` reports 1,
    #     the data path is the critical path (model_ready tracks read at +0.35s), and 10 workers
    #     beat 8 by +36 SPS (+0.8%) over 5 interleaved reps each, t=2.93, 0 predictions changed.
    #     12 and 16 are worse: the data path gets faster but the model-load thread starves. ---
    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 LEAN_WORKERS=10 \
    APP_PORT=7860 \
    # --- explainability (serve-ngrok) knobs: the staged e20 app finds its
    #     preprocessing sources here; device auto-resolves to cuda when the
    #     judge ran --gpus all, cpu otherwise (read by entrypoint.sh) ---
    E20_COMMON_DIR=/app/explain/common \
    E20_E18_DIR=/app/explain/e18_attr_clean \
    EXPLAIN_DEVICE=auto

# ---- 2. Python deps. torch 2.10.0+cu128 is ALREADY in the base image - it is
#         deliberately absent from requirements.txt (a pip torch would re-pull
#         ~3 GB and could swap the CUDA build). Versions pinned to the exact
#         /home/jupyter/trendyol-teknofest-work/.venv-gcp freeze the champion
#         checkpoint was validated with (transformers 4.57.6 stack). ------------
COPY requirements.txt /tmp/build-req/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/build-req/requirements.txt \
 && rm -rf /tmp/build-req \
 && python - <<'PY'
import torch, transformers, tokenizers, safetensors, numpy, pandas
print("import-check:", "torch", torch.__version__, "| transformers", transformers.__version__,
      "| tokenizers", tokenizers.__version__, "| safetensors", safetensors.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
assert torch.__version__.startswith("2.10.0"), torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert tokenizers.__version__ == "0.22.1", tokenizers.__version__
assert safetensors.__version__ == "0.6.2", safetensors.__version__
assert numpy.__version__ == "2.2.4", numpy.__version__
assert pandas.__version__ == "2.3.3", pandas.__version__
# NEVER touch the GPU here: docker build has no GPU and must not need one.
PY

# ---- 2b. ngrok v3 agent: vendored + sha-pinned (explainability tunnel; the
#          runtime may download NOTHING - the only permitted egress is the
#          tunnel itself). Binary staged by prepare_context.sh; sha256 of the
#          static linux-amd64 binary is verified here against vendor/ngrok.sha256
#          (same sha cross-verified from ngrok's apt repo - see README).
#          `ngrok version` runs offline and proves the binary executes. -------
COPY vendor/ngrok.sha256 /app/vendor/ngrok.sha256
COPY vendor/ngrok /app/vendor/ngrok
RUN cd /app/vendor && sha256sum -c ngrok.sha256 && chmod 0755 ngrok && ./ngrok version

# ---- 3. Model weights (~1.2 GB): bigger than src, changes only on a checkpoint
#         swap -> its own layer BEFORE src. prepare_context.sh stages ANY
#         checkpoint dir into models/current/ (rsync -aL, symlinks resolved,
#         chmod -R a+rX on the HOST - never RUN chown here: it would copy the
#         whole tree into a second layer). The in-image path /models/current is
#         FIXED; swapping the checkpoint never edits Dockerfile or scripts. ----
ARG MODEL_CONTEXT_DIR=models/current
COPY ${MODEL_CONTEXT_DIR}/ /models/current/

# ---- 3b. Explainability app (e20) + its preprocessing sources-of-truth.
#          Staged by prepare_context.sh from e20_explain_ty (app + static UI),
#          s2_exp/common (normalize.py T1-T4 + drop_keys.csv) and
#          e18_attr_clean (apply_v2 + drop_keys_v2.csv). The baked
#          E20_COMMON_DIR/E20_E18_DIR point the adapter at these copies;
#          attr_clean.py's own "../common" fallback ALSO resolves correctly in
#          this layout. Served checkpoint = the same /models/current. ----------
COPY explain/ /app/explain/

# ---- 4. Application source. SMALL, changes often -> last. ---------------------
#         src/serve_hf.py + src/lean_infer.py are byte-identical copies of the
#         e12_lean_infer serving pair (staged by prepare_context.sh; sha256 in
#         models/STAGING_MANIFEST.txt). src/predict.py is the thin wrapper that
#         adds CUDA fail-fast + organizer-schema validation around them.
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0755 /app/entrypoint.sh \
 && PYTHONDONTWRITEBYTECODE= python -m compileall -q /app/src /app/explain \
 # bind-mount targets always exist, however an organizer wrapper mounts
 && mkdir -p /input /output \
 && : > /input/input.csv && : > /input/public_test.csv \
 && chmod -R a+rX /app/src /app/explain /app/entrypoint.sh /input /output \
 # sanity: the model layer must be complete and world-readable
 && python - <<'PY'
import json, os, stat
d = "/models/current"
cfg0 = json.load(open(os.path.join(d, "config.json")))
# remote-code files are required only for the GTE "New" family; plain BERT has none
need = ["config.json", "model.safetensors", "tokenizer.json"]
if cfg0.get("auto_map"): need += ["modeling.py", "configuration.py"]
missing = [f for f in need if not os.path.isfile(os.path.join(d, f))]
assert not missing, f"model layer incomplete: missing {missing} - run prepare_context.sh"
bad = [f for f in os.listdir(d) if not os.stat(os.path.join(d, f)).st_mode & stat.S_IROTH]
assert not bad, f"not world-readable (chmod -R a+rX on the HOST before build): {bad}"
cfg = json.load(open(os.path.join(d, "config.json")))
amap = cfg.get("auto_map", {})
for k in ("AutoConfig", "AutoModelForSequenceClassification"):
    assert "--" not in amap.get(k, ""), f"auto_map[{k}]={amap.get(k)} points at the hub; offline load would fail"
print("model-layer-check OK:", cfg.get("architectures"), "num_labels", len(cfg.get("id2label", {})))
PY

WORKDIR /app

# Defence in depth only: the judge's --user overrides this at runtime; nothing
# may DEPEND on uid 10001 (no --create-home: HOME is /tmp/home via ENV).
RUN useradd --no-create-home --uid 10001 --shell /usr/sbin/nologin competitor || true
USER 10001:10001

LABEL org.opencontainers.image.title="ArmaTerimi Trendyol Stage-2 submission (e15 student)" \
      org.opencontainers.image.description="BertForSequenceClassification cross-encoder (BERTurk-128k, 12L, 184.3M), ordinal head (x=p_semi+2*p_rel) with global cuts, lean_infer pipeline" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"

# Exec form is mandatory: the frozen scripts append the positional args.
ENTRYPOINT ["/app/entrypoint.sh"]
