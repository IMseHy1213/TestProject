"""JSON API (FastAPI, async) - v0.1 보완: async/await · 구조화 출력(Pydantic) · 스트리밍(SSE)."""

from .app import create_api

__all__ = ["create_api"]
