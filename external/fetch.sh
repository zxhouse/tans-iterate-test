#!/usr/bin/env bash
# Fetch the deployed coders whose tables and data are re-measured (pinned commits).
set -e
cd "$(dirname "$0")"
get() { [ -d "$1" ] || git clone -q "$2" "$1"; (cd "$1" && git fetch -q --depth 1 origin "$3" 2>/dev/null || true; git checkout -q "$3"); }
get ans-iot       https://github.com/mmr01/ans-iot.git                 cdfb7ec294dd70d72c0b64ad0ff57cca29640fcd
get pivco-huffman https://github.com/MarcinZukowski/pivco-huffman.git  16b48fcfcbd5cd71d945eb48e2702eec398bb36b
get LOCO-ANS      https://github.com/hpcn-uam/LOCO-ANS                 c34e283462d5acbb15bb69fc92c365adccf30fe2
echo "external repositories ready"
