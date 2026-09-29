.PHONY: install dev-api dev-web test eval package-data package-code gen-data package-gen notebooks ingest

API_HOST ?= 127.0.0.1
API_PORT ?= 8000

install:
	pip install -r requirements.txt

dev-api:
	uvicorn server.app:app --host $(API_HOST) --port $(API_PORT) --reload

dev-web:
	cd frontend && npm run dev

test:
	python -m pytest tests model_b_generation/tests -q --tb=short

eval:
	python -m eval.eval_hatecheck
	python -m eval.probe_model

package-data:
	python scripts/package_data.py

# Model B data: converters -> leak-free splits -> artifacts/civitas_gen_data.zip
gen-data:
	python -m model_b_generation.build_dataset_gen
	python -m model_b_generation.prepare_splits_gen $(if $(wildcard data/gen/silver_pairs.parquet),--silver data/gen/silver_pairs.parquet,)

package-gen:
	python scripts/package_data.py --kind gen

# regenerate notebooks/*.ipynb from scripts/build_notebooks.py
notebooks:
	python scripts/build_notebooks.py

# unpack artifacts/civitas_results_*.zip from Colab and print a summary
ingest:
	python scripts/ingest_results.py

# committed code only (no data, weights or node_modules), for Colab notebooks
package-code:
	mkdir -p artifacts
	git archive --format=zip -o artifacts/civitas_code.zip HEAD
