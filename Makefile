
.PHONY: dev-api dev-web test eval install

install:
	pip install -r requirements.txt

dev-api:
	uvicorn server.app:app --host $(API_HOST) --port $(API_PORT) --reload

dev-web:
	cd frontend && npm run dev

test:
	python -m pytest tests/ -q --tb=short

eval:
	python -m eval.run_all

package-data:
	python scripts/package_data.py
