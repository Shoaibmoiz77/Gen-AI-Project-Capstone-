.PHONY: install index test lint eval eval-offline serve docker

install:
	pip install -e ".[dev]"

index:
	groundwork ingest --src data/corpus

test:
	pytest

lint:
	ruff check .

# Full evaluation with Claude as generator and judge (needs ANTHROPIC_API_KEY)
eval: index
	groundwork eval --llm anthropic --out reports/latest

# No API key needed: retrieval metrics plus an extractive baseline
eval-offline: index
	groundwork eval --llm fake --out reports/offline-baseline

serve: index
	groundwork serve --port 8000

docker:
	docker build -t groundwork-rag .
	docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY groundwork-rag
