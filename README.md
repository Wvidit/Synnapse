Just run `start.sh` everthing will setup on its own.

first step:
```
python -m data.fetch_papers
python -m data.build_graph
python -m data.build_taxonomy
python -m data.embed_papers
python -m data.generate_triples
python -m data.generate_qa
```

hosting backend
```
python -m agent.server
```

hosting frontend
```
cd frontend/
npm run dev
```

train model:
```
python -m model.local_grpo
```

benchmarking pipeline:
```
./eval.sh                  # run all 3 stages
./eval.sh --quick          # smoke-test (few samples)
./eval.sh --model-only     # Stage 1: MMLU / BBH / TruthfulQA
./eval.sh --agent-only     # Stage 2: Baseline vs Neurosymbolic
./eval.sh --context-only   # Stage 3: ContextBench
```

individual scripts:
```
python -m eval.benchmark_eval --samples 20       # model benchmarks
python -m eval.agent_benchmark --samples 5       # agent comparison
python -m eval.contextbench_eval --samples 3     # context policies
```

## Evaluation Results

### 1. Model Benchmark (Base vs Fine-Tuned)

| Benchmark | Base Accuracy | Fine-Tuned Accuracy | Delta |
|-----------|---------------|---------------------|-------|
| MMLU | 65.50% | 70.50% | +5.00% |
| Big-Bench Hard | 10.00% | 36.00% | +26.00% |
| TruthfulQA (MC1) | 41.62% | 42.72% | +1.10% |
| Domain QA (Scientific) | 37.91% | 50.38% | +12.47% |

*(Note: Domain QA metric shown is Word Overlap)*

### 2. Agent Benchmark (Baseline vs Neurosymbolic)

| Agent | Accuracy | Hallucination Rate | Avg Latency | Avg Cost/Query |
|-------|----------|--------------------|-------------|----------------|
| Baseline LLM Agent | 80.00% | 37.74% | 0.83s | $0.0005 |
| Neurosymbolic Agent | 100.00% | 12.64% | 0.82s | $0.0006 |

*(Note: The Sample set was pretty small)*

**Per-Task-Type Accuracy:**

| Task Type | Baseline | Neurosymbolic | Delta |
|-----------|----------|---------------|-------|
| hypothesis | 100% | 100% | +0% |
| multi-hop | 50% | 100% | +50% |
| single-hop | 100% | 100% | +0% |
| summarization | 100% | 100% | +0% |
| verification | 50% | 100% | +50% |

### 3. ContextBench (Policy Comparison)

| Policy | Accuracy | Target Budget | Avg Tokens/Query | Avg Cost/Query |
|--------|----------|---------------|------------------|----------------|
| Policy A — Naive (Full Context) | 62.5% | 8192 | 230 | $0.0005 |
| Policy B — RAG Retrieval | 62.5% | 4096 | 255 | $0.0005 |
| Policy C — Compression + Cache | 37.5% | 2048 | 291 | $0.0006 |
