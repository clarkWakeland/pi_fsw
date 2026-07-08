#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/scripts/qcam-dev"

test -x "$script"
bash -n "$script"

help_output="$("$script" --help)"

for expected in "deploy" "revert" "status" "logs"; do
    grep -q "$expected" <<<"$help_output"
done

grep -q "0.0.0-dev" "$script"
