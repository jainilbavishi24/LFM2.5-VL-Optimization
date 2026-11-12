#!/usr/bin/env bash
# Run a command detached on the DGX Spark, in ~/PAA_Team21/PAA_Project with scripts/spark/env.sh loaded.
# From the laptop (master connection open: ssh -fN spark):
#
#   bash scripts/spark/bg.sh <name> "<command>"      # log: ~/PAA_Team21/logs/<name>.log on the Spark
#   ssh spark 'tail -f ~/PAA_Team21/logs/<name>.log'
#
# python = .venv-ref/bin/python (put on PATH by activating the venv). Nothing outside ~/PAA_Team21 is touched.
set -euo pipefail
NAME="$1"; CMD="$2"
printf -v Q '%q' "$CMD"
ssh spark "cd ~/PAA_Team21/PAA_Project && source scripts/spark/env.sh && source .venv-ref/bin/activate && \
    setsid nohup bash -c $Q > ~/PAA_Team21/logs/$NAME.log 2>&1 < /dev/null & echo \"[spark] started $NAME (pid \$!)\""
