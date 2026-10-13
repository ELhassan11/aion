#!/usr/bin/env bash
# تجهيز بيئة العمل المحلية — يُشغّل مرة واحدة: ./scripts/setup.sh
set -euo pipefail

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

[ -f .env ] || cp .env.example .env
echo "Done. Fill in .env, then run: docker compose up --build"
