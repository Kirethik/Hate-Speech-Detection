# Civitas web UI

React + Vite + TypeScript moderator view for the Civitas API.

```bash
npm install
npm run dev          # http://localhost:5173, proxies /api to the FastAPI server on :8000
VITE_MOCK=1 npm run dev   # fixture responses, no backend needed
npm run build
```

Start the backend from the repo root with `make dev-api`.
