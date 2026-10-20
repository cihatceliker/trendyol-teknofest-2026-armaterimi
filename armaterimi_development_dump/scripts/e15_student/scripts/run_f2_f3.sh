#!/usr/bin/env bash
# F2: TY-ecomm-embed (306M, in-domain encoder) and F3: Vision-Flash text tower (Qwen3-0.6B decoder) on the FROZEN e17 set.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen; S=$O/F23_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "F2 TY-ecomm-embed 306M (2 ep, BS 32, LR 2e-5, hard labels)"
TAG=f2_tyecomm MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f2.log 2>&1
tail -1 $O/train_f2.log | cut -c1-160 | tee -a $S
TRUST=1 CKPT=$O/ckpt_f2_tyecomm OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
log "F3 Vision-Flash text tower Qwen3-0.6B (2 ep, BS 32, LR 1e-5, hard labels)"
TAG=f3_visionflash MODEL=$E15/backbones/vision_flash_text BS=32 LR=1e-5 ALPHA=1 BETA=0 DATA=$O/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_f3.log 2>&1
tail -1 $O/train_f3.log | cut -c1-160 | tee -a $S
CKPT=$O/ckpt_f3_visionflash OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
log "F2F3 DONE"
