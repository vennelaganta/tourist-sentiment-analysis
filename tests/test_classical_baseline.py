import pytest

from classical_baseline.train_model import MODEL_NAMES, build_pipeline

X = ["great view loved it", "awful dirty rude", "open at nine", "wonderful stay", "terrible hotel bad", "ticket counter near gate"] * 3
Y = ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"] * 3


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_baseline_pipelines_fit(name):
    pipe = build_pipeline(name, min_df=1, cv=2).fit(X, Y)
    assert pipe.predict(["great wonderful"])[0] in {"Positive", "Negative", "Neutral"}
