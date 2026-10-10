# flake8: noqa
import math

import pytest

import torch

from catalyst.metrics._auc import AUCMetric
from catalyst.metrics.functional._auc import binary_auc


def _compute(scores, targets):
    metric = AUCMetric(compute_per_class_metrics=True)
    metric.update(scores, targets)
    return metric.compute_key_value()


def test_auc_metric_ignores_absent_class() -> None:
    """
    Macro and weighted AUC average over the classes present in the loader only.
    """
    scores = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.6, 0.1],
            [0.6, 0.3, 0.1],
            [0.4, 0.5, 0.1],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])  # class 2 never occurs
    metrics = _compute(scores, targets)

    auc0 = binary_auc(scores[:, 0], (targets == 0).float())[0]
    auc1 = binary_auc(scores[:, 1], (targets == 1).float())[0]
    assert math.isnan(metrics["auc/class_02"])
    assert metrics["auc"] == pytest.approx((auc0 + auc1) / 2)
    assert metrics["auc/_macro"] == pytest.approx((auc0 + auc1) / 2)
    assert metrics["auc/_weighted"] == pytest.approx(0.6 * auc0 + 0.4 * auc1)
    assert not math.isnan(metrics["auc/_micro"])


def test_auc_metric_without_defined_classes() -> None:
    """
    With one class in the loader no per-class AUC is defined, so the averages are nan.
    """
    scores = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    targets = torch.tensor([1, 1, 1])
    metrics = _compute(scores, targets)
    assert math.isnan(metrics["auc"])
    assert math.isnan(metrics["auc/_weighted"])
