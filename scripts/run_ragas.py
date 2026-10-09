"""Score the 50-question set and store the four RAGAS metrics in MLflow."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.eval.metrics import RETRIEVAL_K, load_questions
from egylaw_rag.eval.ragas import (
    RagasScores,
    mean_scores,
    notify_low_faithfulness,
    score_example,
)
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import VectorIndex

EXPERIMENT = "ragas-eval"
_LOG_CHILD = """
import json, sys
from egylaw_rag.tracking import log_ragas_run
payload = json.loads(sys.stdin.read())
print(log_ragas_run(
    payload["uri"],
    payload["experiment"],
    payload["faithfulness"],
    payload["answer_relevancy"],
    payload["context_precision"],
    payload["context_recall"],
    payload["question_count"],
))
"""


def _post_webhook(url: str, payload: dict[str, str]) -> None:
    httpx.post(url, json=payload, timeout=10.0)


def _store_uri(preferred: str, database: Path) -> str:
    """Use the MLflow server when its port is open, otherwise the local sqlite file."""
    if not preferred.startswith("http"):
        return preferred
    parsed = urlparse(preferred)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    with socket.socket() as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            database.parent.mkdir(parents=True, exist_ok=True)
            return "sqlite:///" + database.as_posix()
    return preferred


def _log_run(uri: str, scores: RagasScores, question_count: int, root: Path) -> str:
    """Log in a child process so a stuck server cannot freeze the scorer."""
    payload = json.dumps(
        {
            "uri": uri,
            "experiment": EXPERIMENT,
            "faithfulness": scores.faithfulness,
            "answer_relevancy": scores.answer_relevancy,
            "context_precision": scores.context_precision,
            "context_recall": scores.context_recall,
            "question_count": question_count,
        }
    )
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _LOG_CHILD],
        input=payload,
        capture_output=True,
        text=True,
        timeout=40,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "mlflow log failed")
    return completed.stdout.strip().splitlines()[-1]


def main() -> None:
    settings = get_settings()
    articles = {article.article_number: article for article in read_articles(settings.articles)}
    questions = load_questions(settings.questions_full)
    missing = [
        question.article_number
        for question in questions
        if question.article_number not in articles
    ]
    if missing:
        raise SystemExit(f"corpus is missing evaluation articles: {missing}")
    index = VectorIndex.load(settings.index)
    embedder = LocalEmbedder(settings.embedding_model)
    rows = []
    for question in questions:
        article = articles[question.article_number]
        hits = index.search(embedder.embed_query(question.question_ar), k=RETRIEVAL_K)
        contexts = [hit.text for hit in hits]
        rows.append(
            score_example(
                question.question_ar,
                article.text_normalized,
                contexts,
                [hit.article_number for hit in hits],
                article.article_number,
                article.text_normalized,
            )
        )
    scores = mean_scores(rows)
    print(
        f"faithfulness={scores.faithfulness:.3f} "
        f"answer_relevancy={scores.answer_relevancy:.3f} "
        f"context_precision={scores.context_precision:.3f} "
        f"context_recall={scores.context_recall:.3f}"
    )
    database = settings.project_root / "reports" / "mlflow" / "mlflow.db"
    logged_to = _store_uri(settings.mlflow_tracking_uri, database)
    try:
        run_id = _log_run(logged_to, scores, len(questions), settings.project_root)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        logged_to = "sqlite:///" + database.as_posix()
        database.parent.mkdir(parents=True, exist_ok=True)
        run_id = _log_run(logged_to, scores, len(questions), settings.project_root)
    print(f"logged run {run_id} to {logged_to}")
    if notify_low_faithfulness(scores.faithfulness, settings.alert_webhook_url, _post_webhook):
        print("alert sent")
    if scores.faithfulness < 0.75:
        raise SystemExit(f"faithfulness {scores.faithfulness:.3f} is under 0.75")


if __name__ == "__main__":
    main()
