PYTHON ?= python
export PYTHONPATH := src

.PHONY: all test lint bench up down

all: lint test

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check src tests

bench:
	$(PYTHON) -m hybridsearch benchmark examples/corpus.jsonl examples/queries.jsonl -k 5

up:
	docker compose up -d

down:
	docker compose down -v
