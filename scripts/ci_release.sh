#!/usr/bin/env bash
# Publish the checkpoint to the rolling release, tolerating transient API
# errors. A failed upload leaves the published ingest state ahead of the model,
# so exhausted retries must fail visibly rather than report successful training.
set -uo pipefail

if [ ! -f checkpoint/model.pt ]; then
  echo "no checkpoint to publish"
  exit 0
fi

gh release create model-latest --title "rolling checkpoint" \
  --notes "auto-updated by the training pipeline" 2>/dev/null || true

for attempt in 1 2 3 4; do
  if gh release upload model-latest checkpoint/model.pt --clobber; then
    echo "checkpoint published on attempt ${attempt}"
    exit 0
  fi
  echo "upload attempt ${attempt} failed (transient API error?); retrying"
  sleep $(( attempt * 10 ))
done

echo "::error::checkpoint upload failed after 4 attempts; ingest state and model may be out of step"
exit 1
