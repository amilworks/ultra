#!/bin/bash
set -euo pipefail

cd /source

# Rebuild aggregated UI bundles if a persisted /source/public volume does not
# already contain them.
if [ ! -s /source/public/core/css/all_css.css ] || [ ! -s /source/public/core/js/all_js.js ]; then
  echo "Static bundles missing, running: bq-admin deploy public"
  bq-admin deploy public
fi

bq-admin server start
sleep 2

if [ -f bisque_27000.log ]; then
  exec tail -F bisque_8080.log bisque_27000.log
fi

exec tail -F bisque_8080.log
