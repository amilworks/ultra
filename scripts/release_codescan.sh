#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RG_COMMON=(
  --hidden
  -n
  -I
  --glob '!.git'
  --glob '!**/.venv/**'
  --glob '!**/venv/**'
  --glob '!**/__pycache__/**'
  --glob '!**/node_modules/**'
  --glob '!**/dist/**'
  --glob '!**/build/**'
  --glob '!**/*.ipynb'
  --glob '!**/*.der'
  --glob '!**/*.crt'
  --glob '!**/*.pem'
  --glob '!scripts/release_codescan.sh'
  --glob '!platform/bisque/source/legacy_upgraded/**'
  --glob '!platform/bisque/source/tools/obsolete/**'
  --glob '!platform/bisque/source/bqcore/bq/core/public/js/volume/dagre-d3.js'
)

CHECK_LABELS=(
  "Private key blocks"
  "AWS access keys"
  "GitHub tokens"
  "Slack tokens"
  "Google API keys"
  "OpenAI-style keys"
  "AWS API Gateway URLs"
  "Absolute user paths"
  "Internal UCSB domains"
  "Internal bare hostnames"
)

CHECK_PATTERNS=(
  '-----BEGIN (EC |RSA |OPENSSH )?PRIVATE KEY-----'
  '\b(AKIA|ASIA)[0-9A-Z]{16}\b'
  '\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,})\b'
  '\bxox[baprs]-[A-Za-z0-9-]{10,}\b'
  '\bAIza[0-9A-Za-z_-]{35}\b'
  '\bsk-[A-Za-z0-9]{20,}\b'
  'execute-api\.[A-Za-z0-9-]+\.amazonaws\.com'
  '/Users/[^[:space:]"'"'"'`]+'
  '\b[a-z0-9.-]+\.ece\.ucsb\.edu\b'
  '\b(vrl-h200|nail0[0-9])\b'
)

status=0

for i in "${!CHECK_LABELS[@]}"; do
  label="${CHECK_LABELS[$i]}"
  pattern="${CHECK_PATTERNS[$i]}"
  if matches="$(rg "${RG_COMMON[@]}" -e "${pattern}" . || true)" && [[ -n "${matches}" ]]; then
    printf '\n[%s]\n%s\n' "${label}" "${matches}"
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "release_codescan: clean"
  exit 0
fi

echo "release_codescan: issues found" >&2
exit 1
