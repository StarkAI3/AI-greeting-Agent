#!/usr/bin/env bash
set -euo pipefail

echo "Listening status for ports 6001 and 6002:"
ss -ltnp | grep -E ':6001|:6002' || echo 'no listeners'


