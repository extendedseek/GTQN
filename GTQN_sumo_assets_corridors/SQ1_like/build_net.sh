#!/usr/bin/env bash
set -euo pipefail
netconvert --node-files SQ1_like.nod.xml --edge-files SQ1_like.edg.xml \
  --tls.guess --tls.guess.threshold 0 \
  --default.lanenumber 1 --default.speed 13.90 \
  --output-file SQ1_like.net.xml
echo "Built SQ1_like.net.xml"
