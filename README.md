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

evaluate model:
`
python -m eval.agent_eval
python -m eval.evaluate_model
`

to check per token costing:
`
python -m context.contextbench
`

