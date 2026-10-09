#!/bin/bash
fail() { echo "unexpected $1"; echo "0.0" > /logs/verifier/reward.txt; exit 0; }
[ "$GREETING" = "hello" ] || fail "GREETING=$GREETING"
[ "$JUDGE_KEY" = "judge-secret" ] || fail "JUDGE_KEY=$JUDGE_KEY"
[ "$EMBEDDED" = 'Bearer ${HARBOR_JUDGE_KEY}' ] || fail "EMBEDDED=$EMBEDDED"
[ "$VERIFIER_KEY" = "judge-secret" ] || fail "VERIFIER_KEY=$VERIFIER_KEY"
[ "${EMPTY_DEFAULT-unset}" = "" ] || fail "EMPTY_DEFAULT=${EMPTY_DEFAULT-unset}"
echo "1.0" > /logs/verifier/reward.txt
