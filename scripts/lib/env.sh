#!/usr/bin/env bash
# Source this from the repo root: loads .env if present, stripping CRLF line endings.
# .env format: KEY=value, no quotes, no spaces around "=", values without "$" or "#".
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1090
  . <(sed 's/\r$//' .env)
  set +a
fi
