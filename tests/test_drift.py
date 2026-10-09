import numpy as np

from egylaw_rag.observe.drift import baseline_vector, cosine_similarity


def test_cosine_against_the_index_baseline() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    baseline = baseline_vector(vectors)
    assert cosine_similarity(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 0.0
    assert abs(cosine_similarity(baseline, baseline) - 1.0) < 1e-5
