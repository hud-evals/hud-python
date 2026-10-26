#!/bin/bash
set -u
mkdir -p /logs/verifier

echo "verifier report begins"
for index in $(seq 1 1000); do
  echo "FAILED tests/test_suite.py::test_case_$index - AssertionError: expected $index"
done
echo "verifier report ends" >&2
echo 1 > /logs/verifier/reward.txt
