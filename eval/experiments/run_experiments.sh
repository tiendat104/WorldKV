#!/usr/bin/env bash
# Run the offline-schedule experiments on a 4x A40 pod prepared with eval/setup_pod.sh.
#
# Usage on the pod (all experiments, or some of them, in this order):
#   bash /workspace/WorldKV/eval/experiments/run_experiments.sh [e1] [e2] [e3]
# Unattended, from the laptop (the log is what a watcher waits on):
#   ssh <pod> 'cd /workspace; setsid nohup bash WorldKV/eval/experiments/run_experiments.sh \
#              > /workspace/experiments.log 2>&1 < /dev/null &'
#
# Results go to $OUT/<experiment>/ (runner output: videos, runs/*.telemetry.json,
# results.csv, summary.csv; for e1 and e3 also checks.txt from eval/check_offline.py).
# Each experiment ends with a line "=== <experiment> done", the whole run with
# "=== all done". Every video runs in its own process (--fresh-worker) and
# without quality metrics (--no-score); see Documents/experiment_design.md.
#
# Settings (environment variables): WORKSPACE (/workspace), REPO ($WORKSPACE/WorldKV),
# OUT ($WORKSPACE/experiments), PYTHON (python), P2P_FLAG (--no-p2p; empty on
# NVLink machines), DRY_RUN=1 (print the runner commands only).

set -u
WS=${WORKSPACE:-/workspace}
REPO=${REPO:-$WS/WorldKV}
OUT=${OUT:-$WS/experiments}
PY=${PYTHON:-python}
P2P_FLAG=${P2P_FLAG---no-p2p}
cd "$REPO"
[ -f "$WS/venv/bin/activate" ] && source "$WS/venv/bin/activate"
mkdir -p "$OUT"
git log -1 --format='%H %s' > "$OUT/code_version.txt" 2>/dev/null || true

COMMON=(--ckpt-dir "$WS/models/lingbot-world-base-cam" --nproc 4 --no-score --fresh-worker)
[ -n "$P2P_FLAG" ] && COMMON+=("$P2P_FLAG")
[ "${DRY_RUN:-0}" = 1 ] && COMMON+=(--dry-run)

run() {  # run <run list> <output folder> <methods>
  "$PY" -m eval.runner --benchmark "eval/experiments/$1" --out "$OUT/$2" --methods "$3" \
    "${COMMON[@]}"
  echo "runner exit $? ($1, $3)"
}

check() {  # check <run list> <output folder> [reference method]: identical videos, numbers
  [ "${DRY_RUN:-0}" = 1 ] && return
  local names
  names=$("$PY" -c "import sys; from eval.runner import load_benchmark, run_name; \
print(' '.join(run_name(e) for e in load_benchmark(sys.argv[1])))" "eval/experiments/$1")
  for name in $names; do
    echo "--- $name"
    if [ $# -ge 3 ]; then
      "$PY" -m eval.check_offline "$OUT/$2" "$name" --reference "$OUT/$2/videos/$3/$name.mp4"
    else
      "$PY" -m eval.check_offline "$OUT/$2" "$name"
    fi
  done > "$OUT/$2/checks.txt" 2>&1
  grep -E "^(---|PASS|FAIL)" "$OUT/$2/checks.txt"
}

e1() {
  run e1.json e1 gpu_only,cpu_only,offline
  check e1.json e1 gpu_only
}

e2() {
  run e2_long.json e2_long gpu_only,cpu_only,offline  # WorldKV (GPU) is expected to run out
  run e2_full_kv.json e2_full_kv full_kv              # the longer lengths are expected to fail
}

e3() {
  run e3.json e3 cpu_only,offline
  check e3.json e3
}

for experiment in "${@:-e1 e2 e3}"; do
  for exp in $experiment; do
    echo "=== $exp start $(date)"
    "$exp"
    echo "=== $exp done $(date)"
  done
done
echo "=== all done $(date)"
