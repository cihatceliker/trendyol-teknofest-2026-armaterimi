#!/usr/bin/env bash
# F4: TY-ecomm-embed with attribute cleaning v2 (e18) — paired against F2 (same data/config, v1 text): DEV 0.7994.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; mkdir -p $O; S=$O/F4_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "build with ATTR_CLEAN=v2"
ATTR_CLEAN=v2 LABELS=$E17/outputs/irrelevant_final.parquet CONTRAST=$E17/outputs/contrast_final.parquet OUT=$O $PY scripts/build_student_data.py 2>&1 | grep -E "ATTR_CLEAN|wrote" | cut -c1-200 | tee -a $S
log "train TY-ecomm-embed (2 ep, BS 32, LR 2e-5) on v2 text"
TAG=f4_tyecomm_v2 MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f4.log 2>&1
tail -1 $O/train_f4.log | cut -c1-160 | tee -a $S
ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_f4_tyecomm_v2 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "ATTR_CLEAN|2-boundary|AUC" | tee -a $S
log "reference F2 (v1 text): 0.7994 | F4 DONE"
