.PHONY: install dev-api dev-web test eval package-data

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
