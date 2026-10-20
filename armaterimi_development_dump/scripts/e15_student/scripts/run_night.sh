#!/usr/bin/env bash
# NIGHT QUEUE 2026-08-23->24 (user-selected, ~10.5h). ATTR_CLEAN=v2 everywhere, hard labels, DEV OOF eval only.
# NO API calls, NO HOLDOUT reads. A failed rung logs and moves on.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/NIGHT_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
run(){ # $1 tag $2 model $3 data $4 epochs $5 bs $6 lr $7 extra_env
  log "== $1 ($2 ep=$4 bs=$5 lr=$6 ${7:-})"
  if env ${7:-} TAG=$1 MODEL=$2 TRUST=1 BS=$5 LR=$6 EPOCHS=$4 ALPHA=1 BETA=0 DATA=$3 OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    env ${7:-} ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | tee -a $S
  else
    log "!! $1 TRAIN FAILED (see train_$1.log)"
  fi
}
TY=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0
FIN=$O/student_train.parquet
log "NIGHT START — refs: tyecomm 2ep 0.8149 / 3ep 0.8217 (HOLDOUT 0.8235) | berturk 2ep 0.8028 | vf plain 0.7660"
# 1. best + extras (30.5k judge-semi/relevant added to the best config)
if [ ! -f "$O/build_best_extras/student_train.parquet" ]; then
  ATTR_CLEAN=v2 LABELS=$E17/outputs/irrelevant_final.parquet CONTRAST=$E17/outputs/labels_contrast_plus_extras.parquet OUT=$O/build_best_extras $PY scripts/build_student_data.py > $O/build_best_extras.log 2>&1
fi
run n1_best_extras $TY $O/build_best_extras/student_train.parquet 3 32 2e-5 ""
# 2. Vision-Flash + sandwich on v2 text
run n2_vf_sandwich $E15/backbones/vision_flash_text $FIN 2 32 2e-5 "SANDWICH=1"
# 3. high-potential backbones
run n3_e5large intfloat/multilingual-e5-large $FIN 2 32 1.5e-5 ""
run n4_mdeberta microsoft/mdeberta-v3-base $FIN 2 32 2e-5 ""
# 4. BERTurk, more epochs
run n5_berturk_4ep dbmdz/bert-base-turkish-128k-cased $FIN 4 64 3e-5 ""
# 5. champion budget probe
run n6_tyecomm_4ep $TY $FIN 4 32 2e-5 ""
log "NIGHT QUEUE DONE"
