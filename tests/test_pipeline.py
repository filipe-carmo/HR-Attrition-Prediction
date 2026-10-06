"""Checks that training preprocessing, saved artifacts and the deployed app agree."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from conftest import PROCESSED, RAW_DATA
from preprocessing import preprocess, selected_features, transform_new_data


@pytest.fixture(scope="module")
def splits():
    return preprocess(RAW_DATA)


def test_preprocess_returns_18_features_and_stratified_splits(splits):
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = splits
    assert list(X_train.columns) == selected_features
    assert (len(X_train), len(X_val), len(X_test)) == (882, 294, 294)
    for y in (y_train, y_val, y_test):
        assert y.mean() == pytest.approx(0.16, abs=0.01)


def test_preprocess_reproduces_saved_test_set(splits):
    X_test = splits[2]
    saved = pd.read_csv(PROCESSED / "X_test.csv")
    np.testing.assert_allclose(X_test.astype(float).values, saved.astype(float).values)


def test_deployment_transform_matches_training_transform(splits, raw_df, app_module):
    """transform_new_data (used by the app) must produce exactly the training features."""
    y = raw_df["Attrition"].map({"Yes": 1, "No": 0})
    _, raw_test = train_test_split(raw_df, stratify=y, test_size=0.2, random_state=42)
    deployed = transform_new_data(raw_test, app_module.scaler, app_module.imputer)
    np.testing.assert_allclose(deployed.astype(float).values, splits[2].astype(float).values, atol=1e-9)


@pytest.mark.parametrize(
    ("model_attr", "threshold_attr", "expected_auc", "min_recall"),
    [("logreg", "THRESHOLD_LOGREG", 0.808, 0.75), ("dt", "THRESHOLD_DT", 0.724, 0.70)],
)
def test_saved_models_keep_reported_performance(
    splits, app_module, model_attr, threshold_attr, expected_auc, min_recall
):
    X_test, y_test = splits[2], splits[5]
    model = getattr(app_module, model_attr)
    proba = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) == pytest.approx(expected_auc, abs=0.005)
    predictions = (proba >= getattr(app_module, threshold_attr)).astype(int)
    assert recall_score(y_test, predictions) >= min_recall
