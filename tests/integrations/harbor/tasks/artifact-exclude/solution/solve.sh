#!/bin/sh
set -eu
mkdir -p /app/outputs/cache/nested /app/outputs/logs
echo keep > /app/outputs/keep.txt
echo junk > /app/outputs/junk.tmp
echo junk > /app/outputs/logs/nested.tmp
echo junk > /app/outputs/cache/nested/blob
ln -s /etc/passwd /app/outputs/cache/link
ln -s missing /app/outputs/logs/dangling.tmp
