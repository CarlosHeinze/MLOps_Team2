#!/usr/bin/env bash
# Runs inside a fresh python:3.12.12-slim-trixie container: clean clone, locked install,
# tests, and the reproducibility comparison of each champion and of the published image.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
apt-get update -qq && apt-get install -y -qq --no-install-recommends git make libgomp1 > /dev/null
git config --global --add safe.directory '*'
git clone -q --branch "$BRANCH" /repo /work
cd /work
echo "== commit $(git rev-parse HEAD)"
python -m venv .venv
start=$(date +%s)
make requirements > /tmp/pip.log 2>&1 || { tail -20 /tmp/pip.log; exit 1; }
echo "== make requirements: $(( $(date +%s) - start )) s, $(du -sh .venv | cut -f1)"
.venv/bin/python -m pip check
mkdir -p data/processed/mibici
cp -r /features data/processed/mibici/features
cp -r /models/. models/
start=$(date +%s)
make test 2>&1 | tail -1
echo "== make test: $(( $(date +%s) - start )) s"
for target in departures arrivals; do
  .venv/bin/python -m team2_mlops2026.modeling.reproducibility compare --no-track \
    --target "$target" --url "$SERVICE_URL" --save "/out/reproduction-$target.json" 2>&1 \
    | grep -E "^(OK|FAIL)"
done
