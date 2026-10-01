#!/bin/bash
# DL robustness: 3 seeds on 2024->2025, then swapped years 2025->2024 (seed 0). Sequential to keep CPU sane.
cd "$(dirname "$0")"
for cfg in "0 2024 2025" "1 2024 2025" "2 2024 2025" "0 2025 2024"; do
  set -- $cfg
  SEED=$1 TRAIN=$2 TEST=$3 "${PYTHON:-python}" model2.py 2>&1 | grep -vE 'Warning|from pandas|warn\(|apply operated|scout = ' > "robust_s$1_$2to$3.txt"
  grep SUMMARY "robust_s$1_$2to$3.txt"
done
echo ROBUST_DONE
