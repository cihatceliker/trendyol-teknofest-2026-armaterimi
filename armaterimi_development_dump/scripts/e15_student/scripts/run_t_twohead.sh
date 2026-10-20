#!/usr/bin/env bash
# T-chain (e21): two-head + Stage-1 head-A data (139,147 rows: ALL unique mix2 pairs minus those
# already in Stage-2 minus public collisions). Baseline: r1_oldstore 3-way, DEV 0.8553, AUC(R|S) 0.9601.
set -u
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5
E21=$W/asya/s2_exp/e21_stage1_headA; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/T_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
TY=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0
build(){ [ -f "$O/$1/student_train.parquet" ] && { log "build $1 cached"; return 0; }
  local extra=""; [ -n "$2" ] && extra="STAGE1=$E21/outputs/stage1_headA.parquet STAGE1_BANDS=$2"
  env ATTR_CLEAN=v2 LABELS=$E17/outputs/labels_allI_plus_old.parquet CONTRAST=$E17/outputs/labels_extras_plus_old.parquet \
    $extra OUT=$O/$1 $PY scripts/build_student_data.py 2>&1 | grep -E "stage1 head-A|wrote" | cut -c1-170 | tee -a $S; }
run(){ n=$(ps -eo cmd | grep -c "[t]rain_student.py"); [ "$n" -gt 0 ] && { log "GUARD"; exit 1; }
  log "== $1"
  if TWO_HEAD=1 TAG=$1 MODEL=$TY TRUST=1 BS=32 LR=2e-5 EPOCHS=3 ALPHA=1 BETA=0 DATA=$2 OUT=$O \
     $PY scripts/train_student.py > $O/train_$1.log 2>&1; then
    ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary|AUC" | tee -a $S
  else log "!! $1 FAILED"; tail -4 $O/train_$1.log | tee -a $S; fi; }
log "T-CHAIN — baseline r1_oldstore 0.8553 (AUC R|S 0.9601, S|I 0.9037)"
build build_2h_s1all confident,uncertain
run t2_twohead_s1all $O/build_2h_s1all/student_train.parquet     # all 139,147 stage-1 head-A rows
build build_2h_s1conf confident
run t3_twohead_s1conf $O/build_2h_s1conf/student_train.parquet   # confident band only (95,885)
build build_oldstore ""
run t1_twohead $O/build_oldstore/student_train.parquet           # architecture control, no stage-1
log "T-CHAIN DONE"
