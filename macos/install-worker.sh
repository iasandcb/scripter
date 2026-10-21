#!/bin/sh
# Install (or remove) the login LaunchAgent that polls the mark-vector queue.
# Secrets stay in the repo .env. This script does not write them into the plist.
set -eu
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 필요합니다. brew install uv" >&2
  exit 1
fi
if [ "${1:-}" = "--uninstall" ]; then
  exec uv run python -c 'import worker; worker.uninstall_launch_agent()'
fi
if [ "${1:-}" != "" ]; then
  echo "사용법: $0 [--uninstall]" >&2
  exit 1
fi
exec uv run python -c 'import worker; worker.install_launch_agent()'
