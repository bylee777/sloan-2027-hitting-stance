#!/bin/bash
# Perception network with POSTURE-BASED EYE HEIGHT: same 4 configs as run_robust.sh (+ a trunk-lean sensitivity), paired with the original runs.
cd "$(dirname "$0")"
for cfg in "0 2024 2025 1" "1 2024 2025 1" "2 2024 2025 1" "0 2025 2024 1" "0 2024 2025 2"; do
  set -- $cfg
  EYE=posture EYE_LEAN=$4 SEED=$1 TRAIN=$2 TEST=$3 "${PYTHON:-python}" model2.py 2>&1 | grep -vE 'Warning|from pandas|warn\(|apply operated|scout = ' > "robust_s$1_$2to$3_eye$4.txt"
  grep SUMMARY "robust_s$1_$2to$3_eye$4.txt"
done
echo EYE_ROBUST_DONE
