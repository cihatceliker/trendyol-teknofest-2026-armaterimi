#!/usr/bin/env bash
# F1: BERTurk-128k on the FROZEN e17 set — same config as e1_clean (unjudged v4, DEV 0.7766) for a clean comparison.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen; mkdir -p $O; S=$O/F1_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
log "build (organizer R/S + irrelevant_final + contrast)"
LABELS=$E17/outputs/irrelevant_final.parquet CONTRAST=$E17/outputs/contrast_final.parquet OUT=$O $PY scripts/build_student_data.py 2>&1 | grep -E "organizer|judged-I|contrast|wrote" | tee -a $S
log "train BERTurk-128k (2 ep, BS 64, LR 3e-5, hard labels, class-weighted)"
TAG=f1_berturk DATA=$O/student_train.parquet OUT=$O ALPHA=1 BETA=0 $PY scripts/train_student.py > $O/train_f1.log 2>&1
tail -2 $O/train_f1.log | tee -a $S
log "eval (DEV OOF two-boundary)"
CKPT=$O/ckpt_f1_berturk OUT=$O $PY scripts/eval_student.py 2>&1 | tail -8 | tee -a $S
$PY -c "import json;d=json.load(open('$O/eval_f1_berturk.json'));t=d['two_boundary_oof'];print('F1 BERTurk frozen-data two-boundary OOF', round(t['macro_f1'],4), {k:round(v,3) for k,v in t['per_class'].items()}, '| reference e1_clean (unjudged v4) 0.7766')" | tee -a $S
log "F1 DONE"
