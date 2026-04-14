#!/bin/bash
# ============================================
#   Synnapse — Evaluation Pipeline
#   Runs all evaluation scripts:
#     1. Agent end-to-end eval
#     2. Domain QA (base vs fine-tuned)
#     3. Benchmarks: MMLU, Big-Bench Hard,
#        TruthfulQA
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🧪  Synnapse Evaluation Suite      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# 1. Activate virtual environment
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found. Using system Python."
fi

cd "$PROJECT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────
SKIP_AGENT=false
SKIP_DOMAIN=false
SKIP_BENCH=false
BENCH_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-agent)   SKIP_AGENT=true; shift ;;
        --skip-domain)  SKIP_DOMAIN=true; shift ;;
        --skip-bench)   SKIP_BENCH=true; shift ;;
        --bench-only)   SKIP_AGENT=true; SKIP_DOMAIN=true; shift ;;
        --quick)        BENCH_ARGS="--samples 20"; shift ;;
        --full)         BENCH_ARGS="--full"; shift ;;
        --samples)      BENCH_ARGS="--samples $2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-agent    Skip agent end-to-end eval"
            echo "  --skip-domain   Skip domain QA eval"
            echo "  --skip-bench    Skip benchmark eval (MMLU/BBH/TruthfulQA)"
            echo "  --bench-only    Run only benchmarks"
            echo "  --quick         Run benchmarks on 20 samples (smoke test)"
            echo "  --full          Run benchmarks on full datasets"
            echo "  --samples N     Run benchmarks on N samples per benchmark"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
SKIP=0

run_eval() {
    local name="$1"
    local cmd="$2"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📊 $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$cmd"; then
        echo "  ✅ $name — PASSED"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $name — FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Agent End-to-End Eval ─────────────────────────────────────────────────
if [ "$SKIP_AGENT" = false ]; then
    run_eval "Agent End-to-End Eval" "python -m eval.agent_eval"
else
    echo "⏭️  Skipping Agent Eval"
    SKIP=$((SKIP + 1))
fi

# ── 2. Domain QA Eval ────────────────────────────────────────────────────────
if [ "$SKIP_DOMAIN" = false ]; then
    run_eval "Domain QA Eval (Base vs Fine-Tuned)" "python -m eval.evaluate_model"
else
    echo "⏭️  Skipping Domain QA Eval"
    SKIP=$((SKIP + 1))
fi

# ── 3. Benchmark Eval (MMLU, BBH, TruthfulQA) ───────────────────────────────
if [ "$SKIP_BENCH" = false ]; then
    run_eval "Benchmarks (MMLU / BBH / TruthfulQA)" "python -m eval.benchmark_eval $BENCH_ARGS"
else
    echo "⏭️  Skipping Benchmark Eval"
    SKIP=$((SKIP + 1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   📋  Evaluation Summary             ║"
echo "  ╠══════════════════════════════════════╣"
echo "  ║   ✅ Passed:  $PASS                          ║"
echo "  ║   ❌ Failed:  $FAIL                          ║"
echo "  ║   ⏭️  Skipped: $SKIP                          ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  ⚠️  Some evaluations failed. Check the logs above."
    exit 1
else
    echo "  🎉 All evaluations completed successfully!"
fi
