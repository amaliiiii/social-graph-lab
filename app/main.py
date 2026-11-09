"""Веб-сервис «Лаборатория графов взаимодействий» (FastAPI).

Без базы данных и без авторизации: состояние сервера — только кэш в памяти.
Запуск:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sgl import service

app = FastAPI(title="Лаборатория графов взаимодействий", version="1.0.0")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Source(BaseModel):
    dataset: str | None = None
    edges_text: str | None = Field(default=None, max_length=8_000_000)
    directed: bool = True


class AnalyzeRequest(BaseModel):
    source: Source
    max_nodes: int = Field(default=150, ge=10, le=400)


class CascadeRequest(BaseModel):
    source: Source
    seeds: list[str]
    model: str = "ic"
    p: float = 0.1
    run: int = 0
    max_nodes: int = Field(default=150, ge=10, le=400)


class CompareRequest(BaseModel):
    source: Source
    metric: str
    k: int = 5
    model: str = "ic"
    p: float = 0.1
    runs: int = 100


class QuizRequest(BaseModel):
    source: Source
    node: str
    metric: str
    guess: float


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:                    # ошибки ввода пользователя
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sources")
def sources():
    return service.list_sources()


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    return _call(service.analyze, req.source.model_dump(), req.max_nodes)


@app.post("/api/cascade")
def cascade(req: CascadeRequest):
    return _call(service.cascade, req.source.model_dump(), req.seeds, req.model,
                 req.p, req.run, req.max_nodes)


@app.post("/api/compare")
def compare(req: CompareRequest):
    return _call(service.compare, req.source.model_dump(), req.metric, req.k,
                 req.model, req.p, req.runs)


@app.post("/api/quiz")
def quiz(req: QuizRequest):
    return _call(service.quiz, req.source.model_dump(), req.node, req.metric, req.guess)


# статические файлы подключаются последними, чтобы не перекрывать /api
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
