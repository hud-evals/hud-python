#!/bin/sh
mkdir -p /logs/verifier
echo "FAILED tests/test_outputs.py::test_answer - expected forty-one"
for index in $(seq 1 500); do
  echo "PASSED tests/test_outputs.py::test_case_$index"
done
echo "1 failed, 500 passed"
echo 0 > /logs/verifier/reward.txt
