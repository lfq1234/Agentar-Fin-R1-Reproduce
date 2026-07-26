"""FastAPI entrypoint for the Agentar-Fin-R1 reproduction service.

Exposes:
- GET  /health        liveness probe
- POST /v1/chat       chat completion (delegates to the agent runtime)
- POST /v1/analyze    financial task analysis (intent / slot / tool-plan)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat

app = FastAPI(title="Agentar-Fin-R1 Reproduce API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
