#!/bin/bash
# ============================================
#   Synnapse — Evaluation Pipeline
#   4-Stage Benchmarking Suite:
#     1. Model Benchmark   (MMLU / BBH / TruthfulQA)
#     2. Agent Benchmark   (Baseline vs Neurosymbolic)
#     3. ContextBench      (Naive / RAG / Compression)
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   🧪  Synnapse Benchmarking Pipeline         ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── Activate venv ────────────────────────────────────────────────────────────
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found. Using system Python."
fi

cd "$PROJECT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────
RUN_MODEL=true
RUN_AGENT=true
RUN_CTXBENCH=true
BENCH_ARGS=""
AGENT_ARGS=""
CTX_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-only)    RUN_AGENT=false; RUN_CTXBENCH=false; shift ;;
        --agent-only)    RUN_MODEL=false; RUN_CTXBENCH=false; shift ;;
        --context-only)  RUN_MODEL=false; RUN_AGENT=false; shift ;;
        --skip-model)    RUN_MODEL=false; shift ;;
        --skip-agent)    RUN_AGENT=false; shift ;;
        --skip-context)  RUN_CTXBENCH=false; shift ;;
        --quick)         BENCH_ARGS="--samples 10"; AGENT_ARGS="--samples 3"; CTX_ARGS="--samples 3"; shift ;;
        --full)          BENCH_ARGS="--full"; shift ;;
        --samples)       BENCH_ARGS="--samples $2"; AGENT_ARGS="--samples $2"; CTX_ARGS="--samples $2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./eval.sh [OPTIONS]"
            echo ""
            echo "Stages:"
            echo "  --model-only     Run only model benchmarks (MMLU/BBH/TruthfulQA)"
            echo "  --agent-only     Run only agent benchmarks (requires server)"
            echo "  --context-only   Run only ContextBench (requires server)"
            echo "  --skip-model     Skip model benchmarks"
            echo "  --skip-agent     Skip agent benchmarks"
            echo "  --skip-context   Skip ContextBench"
            echo ""
            echo "Sampling:"
            echo "  --quick          Smoke-test (10 model, 3 agent, 3 context)"
            echo "  --full           Full datasets for model benchmarks"
            echo "  --samples N      N samples per benchmark stage"
            echo ""
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
SKIP=0

run_stage() {
    local name="$1"
    local cmd="$2"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📊 $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$cmd"; then
        echo "  ✅ $name — PASSED"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $name — FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Stage 1: Model Benchmarks ───────────────────────────────────────────────
if [ "$RUN_MODEL" = true ]; then
    run_stage "Stage 1 — Model Benchmark (MMLU / BBH / TruthfulQA)" \
              "python -m eval.benchmark_eval $BENCH_ARGS"
else
    echo "⏭️  Skipping Stage 1 (Model Benchmark)"
    SKIP=$((SKIP + 1))
fi

# ── Stage 2: Agent Benchmarks ───────────────────────────────────────────────
if [ "$RUN_AGENT" = true ]; then
    run_stage "Stage 2 — Agent Benchmark (Baseline vs Neurosymbolic)" \
              "python -m eval.agent_benchmark $AGENT_ARGS"
else
    echo "⏭️  Skipping Stage 2 (Agent Benchmark)"
    SKIP=$((SKIP + 1))
fi

# ── Stage 3: ContextBench ───────────────────────────────────────────────────
if [ "$RUN_CTXBENCH" = true ]; then
    run_stage "Stage 3 — ContextBench (Context Management)" \
              "python -m eval.contextbench_eval $CTX_ARGS"
else
    echo "⏭️  Skipping Stage 3 (ContextBench)"
    SKIP=$((SKIP + 1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   📋  Pipeline Summary                       ║"
echo "  ╠══════════════════════════════════════════════╣"
printf "  ║   ✅ Passed:  %-30s ║\n" "$PASS"
printf "  ║   ❌ Failed:  %-30s ║\n" "$FAIL"
printf "  ║   ⏭️  Skipped: %-30s ║\n" "$SKIP"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
echo "  Result files:"
echo "    eval/benchmark_results.json"
echo "    eval/agent_benchmark_results.json"
echo "    eval/contextbench_results.json"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  ⚠️  Some stages failed. Check logs above."
    exit 1
else
    echo "  🎉 All stages completed successfully!"
fi
