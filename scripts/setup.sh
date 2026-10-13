#!/usr/bin/env bash
# Set up the local dev environment - run once: ./scripts/setup.sh
set -euo pipefail

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

[ -f .env ] || cp .env.example .env
echo "Done. Fill in .env, then run: docker compose up --build"
