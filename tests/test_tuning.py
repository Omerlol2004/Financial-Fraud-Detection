import pandas as pd

from src.tune import TUNING_VALIDATION_SIZE, create_tuning_validation_split


def test_tuning_validation_split_is_disjoint_and_stratified():
    features = pd.DataFrame({"feature": range(100)})
    target = pd.Series([0] * 90 + [1] * 10)

    x_inner, x_validation, y_inner, y_validation = create_tuning_validation_split(features, target)

    assert len(x_validation) == int(len(features) * TUNING_VALIDATION_SIZE)
    assert set(x_inner.index).isdisjoint(set(x_validation.index))
    assert y_inner.sum() == 8
    assert y_validation.sum() == 2
