#!/usr/bin/env bash
# G-ladder (all TY-ecomm-embed, v2 text, 2 ep unless stated; reference F4 = 0.8149 on final 143.9k):
# G1 all judged-I (final+reserve) | G2 +extras (quarantined judge-S/R) | G3 dose 100k | G4 dose 50k | G5 4 epochs on final.
set -uo pipefail
W=/home/jupyter/trendyol-teknofest-work; E15=$W/asya/s2_exp/e15_student; E17=$W/asya/s2_exp/e17_mine_v5; PY=$W/.venv-gcp/bin/python
O=$E15/outputs_frozen_v2; S=$O/G_SUMMARY.txt; : > $S
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $S; }
cd $E15
run(){ # $1 tag  $2 labels  $3 contrast  $4 epochs
  log "== $1 (labels=$(basename $2), contrast=$(basename $3), ep=$4)"
  ATTR_CLEAN=v2 LABELS=$2 CONTRAST=$3 OUT=$O/build_$1 $PY scripts/build_student_data.py 2>&1 | grep -E "wrote" | cut -c1-170 | tee -a $S
  TAG=$1 MODEL=Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 TRUST=1 BS=32 LR=2e-5 EPOCHS=$4 ALPHA=1 BETA=0 \
    DATA=$O/build_$1/student_train.parquet OUT=$O $PY scripts/train_student.py > $O/train_$1.log 2>&1
  ATTR_CLEAN=v2 TRUST=1 CKPT=$O/ckpt_$1 OUT=$O $PY scripts/eval_student.py 2>&1 | grep -E "2-boundary" | tee -a $S
}
run g1_allI      $E17/outputs/labels_allI.parquet              $E17/outputs/contrast_final.parquet              2
run g2_extras    $E17/outputs/labels_allI.parquet              $E17/outputs/labels_contrast_plus_extras.parquet 2
run g3_dose100k  $E17/outputs/labels_dose100k.parquet          $E17/outputs/contrast_final.parquet              2
run g4_dose50k   $E17/outputs/labels_dose50k.parquet           $E17/outputs/contrast_final.parquet              2
run g5_4ep       $E17/outputs/irrelevant_final.parquet         $E17/outputs/contrast_final.parquet              4
log "G LADDER DONE"
