Just run `start.sh` everthing will setup on its own.

first step:
`
python -m data.fetch_papers
python -m data.build_graph
python -m data.build_taxonomy
python -m data.embed_papers
python -m data.generate_triples
python -m data.generate_qa
`

hosting backend
`
python -m agent.server
`

hosting frontend
`
cd frontend/
npm run dev
`

train model:
`
python -m model.train_lora
`

benchmarking pipeline:
`
./eval.sh                  # run all 3 stages
./eval.sh --quick          # smoke-test (few samples)
./eval.sh --model-only     # Stage 1: MMLU / BBH / TruthfulQA
./eval.sh --agent-only     # Stage 2: Baseline vs Neurosymbolic
./eval.sh --context-only   # Stage 3: ContextBench
`

individual scripts:
`
python -m eval.benchmark_eval --samples 20       # model benchmarks
python -m eval.agent_benchmark --samples 5       # agent comparison
python -m eval.contextbench_eval --samples 3     # context policies
`
