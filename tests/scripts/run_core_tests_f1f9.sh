#!/bin/bash
# F1-F9 Core Framework Kernel Tests
# Tests bug fixes and kernel correctness

MUMAX3="${HOME}/go/bin/mumax3"
LOGDIR="tests/core"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/f1f9_validation_${TIMESTAMP}.log"

echo "=========================================" | tee "$LOG"
echo "F1-F9 Framework Kernel Validation" | tee -a "$LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG"
echo "=========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

PASS=0
FAIL=0

# Run each F1-F9 test
for i in {1..9}; do
    TEST="tests/core/test_f${i}_*.mx3"
    if ls $TEST 1> /dev/null 2>&1; then
        TESTFILE=$(ls $TEST)
        TESTNAME=$(basename "$TESTFILE" .mx3)
        
        echo "Running F${i}: $TESTNAME..." | tee -a "$LOG"
        
        if $MUMAX3 "$TESTFILE" > /dev/null 2>&1; then
            echo "  ✅ PASS" | tee -a "$LOG"
            ((PASS++))
        else
            echo "  ❌ FAIL" | tee -a "$LOG"
            ((FAIL++))
        fi
    else
        echo "F${i}: Test file not found" | tee -a "$LOG"
        ((FAIL++))
    fi
done

echo "" | tee -a "$LOG"
echo "=========================================" | tee -a "$LOG"
echo "Results: $PASS passed, $FAIL failed" | tee -a "$LOG"
echo "=========================================" | tee -a "$LOG"

if [ $FAIL -eq 0 ]; then
    echo "✅ ALL F1-F9 TESTS PASSED" | tee -a "$LOG"
    exit 0
else
    echo "❌ SOME TESTS FAILED" | tee -a "$LOG"
    exit 1
fi
