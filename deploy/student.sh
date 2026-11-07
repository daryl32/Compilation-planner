#!/usr/bin/env bash
# The student tag model on the server (CPU) — same commands as the laptop's student_tags.bat:
#
#   deploy/student.sh setup                 one-off: CPU PyTorch + numpy into /root/modal-venv
#   deploy/student.sh fetch                 new / changed video data from Modal (free) + your labels
#   deploy/student.sh all --name v13        fetch, train, score
#   deploy/student.sh rescore-all           every saved model on today's test labels
#   deploy/student.sh picks --name v13      🎓 Student's picks for 🎯 Training Labels
#   deploy/student.sh export --name v13     scores for every half second → catalogue/student_scores
#
# Runs at low priority with a memory cap, so the live app always comes first, and logs to
# /root/data/logs/student.log. Long runs: start it inside tmux (tmux new -A -s student) and detach.
# Data lives on the Hetzner Volume (<volume>/student/_student); results the app reads (runs, picks,
# scores) go straight into /root/data/catalogue — no Drive sync needed for the live app.
set -uo pipefail

APP=$(cd "$(dirname "$0")/.." && pwd)     # the live app folder (this script is in its deploy/)
PY=${STUDENT_PY:-/root/modal-venv/bin/python}
VOL=$(ls -d /mnt/HC_Volume_* 2>/dev/null | head -1)
export STUDENT_WORK_DIR=${STUDENT_WORK_DIR:-${VOL:+$VOL/student}}
export STUDENT_WORK_DIR=${STUDENT_WORK_DIR:-/root/data/student}
THREADS=${STUDENT_THREADS:-3}            # leave a core for the app (the server has 4)
MEM=${STUDENT_MEM:-4G}                   # the run is stopped past this, never the app
LOG=${STUDENT_LOG:-/root/data/logs/student.log}

if [ "${1:-}" = "setup" ]; then
  echo "Installing CPU PyTorch + numpy into $(dirname "$(dirname "$PY")") (about 700 MB on disk)…"
  "$(dirname "$PY")/pip" install --upgrade numpy
  "$(dirname "$PY")/pip" install torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -c "import torch, numpy, modal; print('OK — torch', torch.__version__, '· numpy', numpy.__version__, '· modal', modal.__version__)"
  mkdir -p "$STUDENT_WORK_DIR"
  echo "Student data folder: $STUDENT_WORK_DIR"
  exit 0
fi

if [ $# -eq 0 ]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

mkdir -p "$STUDENT_WORK_DIR" "$(dirname "$LOG")"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "=== $(date '+%F %T') student $* (data: $STUDENT_WORK_DIR, $THREADS threads, max $MEM)" | tee -a "$LOG"
START=$(date +%s)
cd "$APP" || exit 1
if command -v systemd-run >/dev/null 2>&1 && systemd-run --scope --quiet true >/dev/null 2>&1; then
  systemd-run --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    nice -n 10 "$PY" -u student_tags.py "$@" 2>&1 | tee -a "$LOG"
else                                     # no systemd (a container): no memory cap, still low priority
  nice -n 10 "$PY" -u student_tags.py "$@" 2>&1 | tee -a "$LOG"
fi
RC=${PIPESTATUS[0]}
echo "=== done in $(( ($(date +%s) - START) / 60 )) min, exit $RC" | tee -a "$LOG"
exit "$RC"
