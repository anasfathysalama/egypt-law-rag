from egylaw_rag.eval.ragas import (
    ALERT_BELOW,
    answer_relevancy,
    context_precision,
    context_recall,
    mean_scores,
    notify_low_faithfulness,
    ragas_faithfulness,
    score_example,
)
from egylaw_rag.tracking import log_ragas_run


def test_four_ragas_scores() -> None:
    scores = score_example(
        question="contract parties",
        answer="The contract binds the parties.",
        contexts=["The contract binds the parties under the code."],
        retrieved=[147, 10],
        gold=147,
        reference="The contract binds the parties under the code.",
    )
    assert scores.faithfulness == 1.0
    assert scores.answer_relevancy == 1.0
    assert scores.context_precision == 1.0
    assert scores.context_recall == 1.0
    assert ragas_faithfulness("unrelated verdict", ["contract text"]) == 0.0
    assert answer_relevancy("contract", "nothing") == 0.0
    assert context_precision([10, 147], 147) == 0.5
    assert context_recall("contract parties", ["contract"]) == 0.5


def test_alert_fires_under_0_80() -> None:
    sent: list[tuple[str, dict[str, str]]] = []

    def post(url: str, payload: dict[str, str]) -> None:
        sent.append((url, payload))

    assert ALERT_BELOW == 0.80
    assert notify_low_faithfulness(0.79, "http://hook", post) is True
    assert "0.790" in sent[0][1]["content"]
    assert notify_low_faithfulness(0.80, "http://hook", post) is False
    assert notify_low_faithfulness(0.10, "", post) is False
    assert len(sent) == 1


def test_mean_scores_and_mlflow_trend(tmp_path) -> None:
    import mlflow

    first = score_example(
        "contract",
        "contract binds",
        ["contract binds"],
        [1],
        1,
        "contract binds",
    )
    second = score_example("sale", "sale price", ["sale price"], [2], 2, "sale price")
    mean = mean_scores([first, second])
    assert mean.faithfulness == 1.0
    tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    log_ragas_run(tracking_uri, "ragas-eval", 0.9, 0.8, 0.7, 0.6, 20)
    log_ragas_run(tracking_uri, "ragas-eval", 0.91, 0.81, 0.71, 0.61, 50)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("ragas-eval")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 2
    assert set(runs["metrics.faithfulness"]) == {0.9, 0.91}
