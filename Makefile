.PHONY: install models run test lint bench docker

install:
	pip install -e ".[dev,bench,client]"

models:
	bash scripts/download_models.sh --bench

run:
	rtspeech-server --port 8000

test:
	pytest -q

lint:
	ruff check src tests bench clients

bench:
	python bench/make_conversations.py --n 4 --seed 1 --out bench/dev
	python bench/make_conversations.py --n 12 --seed 7 --out bench/data
	python bench/tune.py --data bench/dev
	python bench/run_benchmark.py --data bench/data

docker:
	docker build -t streaming-asr-diarization .
	docker run --rm -p 8000:8000 streaming-asr-diarization
