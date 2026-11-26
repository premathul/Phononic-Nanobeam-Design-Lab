#!/usr/bin/env bash
# Athul Prem | _AP_pnc. Run from the pinned solver checkout root.
set -euo pipefail
pkg="$(realpath "$1")"
out="$(realpath -m "$2")"
mkdir -p "$out"
export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
for spec in 225 168p75; do
  python tools/map_scattering_AP_pnc.py --config "$pkg/config_${spec}_AP_pnc.json" --frequencies-ghz 6 --output "$out/edge${spec}_6GHz_AP_pnc" --max-new-points 1
done
for n in 1 2 3 4; do
  python tools/seed_frequency_map_AP_pnc.py --from-map "$out/edge225_6GHz_AP_pnc" --output "$out/edge225_expansion_AP_pnc" --frequencies-ghz 2 4 5.8 6.2 --max-new-points 1
done
mkdir -p "$out/report_AP_pnc"
MPLBACKEND=Agg python "$pkg/report_AP_pnc.py" "$out" "$out/report_AP_pnc"
python "$pkg/fixed_launch_AP_pnc.py" "$out" "$out/report_AP_pnc/fixed_6GHz_launch_AP_pnc.json"
