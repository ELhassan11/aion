# Aion Inbox UI (wired to the real backend)

React + Vite inbox, connected to AION's real pipeline via the Python API
server (`frontend/serve.py`).

## What is real now
- Conversation list, messages, and customer profiles come from
  `GET /api/inbox` and `GET /api/inbox/<ticket-id>` (tickets +
  customers + Customer 360 profiles).
- **Send** posts a reply to `POST /api/inbox/<id>/reply` (persisted in
  `out/inbox_runtime.json`).
- **AI Reply** calls `POST /api/inbox/<id>/ai-reply` — the real
  customer service agent (retrieval + cited Arabic answer + verification
  + risk check). The UI shows whether the reply was verified and how
  many citations it carries.
- **Close** marks the ticket resolved via `POST /api/inbox/<id>/close`.

## Run
1. In the repo root, run the API + dashboard server:
   ```bash
   python -m frontend.serve --port 8000
   ```
2. Build the UI (needs Node.js 18+):
   ```bash
   cd frontend/inbox-ui
   npm install
   npm run build
   ```
3. Open http://localhost:8000/inbox/ — the Python server serves the
   built app and the API from the same origin (no CORS issues).

For UI development, `npm run dev` (Vite on :5173) works too — the API
sends `Access-Control-Allow-Origin: *`.

## Notes
- Notes in the customer sidebar and the AI toggles are still local UI
  state (demo); everything else hits the real backend.
- Runtime messages are appended to `out/inbox_runtime.json`; the sample
  data files are never modified.
