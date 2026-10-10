# flake8: noqa
import math
import warnings

import numpy as np

import torch

from catalyst.metrics.functional._auc import auc, binary_auc


def test_auc():
    """
    Tests for catalyst.metrics.auc metric.
    """
    test_size = 1000
    scores = torch.cat((torch.rand(test_size), torch.rand(test_size)))
    targets = torch.cat((torch.zeros(test_size), torch.ones(test_size)))

    val = auc(scores, targets)
    assert math.fabs(val - 0.5) < 0.1, "AUC test1 failed"

    scores = torch.cat(
        (
            torch.Tensor(test_size).fill_(0),
            torch.Tensor(test_size).fill_(0.1),
            torch.Tensor(test_size).fill_(0.2),
            torch.Tensor(test_size).fill_(0.3),
            torch.Tensor(test_size).fill_(0.4),
            torch.ones(test_size),
        )
    )
    targets = torch.cat(
        (
            torch.zeros(test_size),
            torch.zeros(test_size),
            torch.zeros(test_size),
            torch.zeros(test_size),
            torch.zeros(test_size),
            torch.ones(test_size),
        )
    )

    val = auc(scores, targets)
    assert math.fabs(val - 1.0) < 0.0001, "AUC test2 failed"


def test_binary_auc_single_class() -> None:
    """
    A single-class target column has an undefined (nan) AUC and raises no warning.
    """
    scores = torch.tensor([0.9, 0.2, 0.7, 0.4])
    for targets in (torch.zeros(4), torch.ones(4)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value, _, _ = binary_auc(scores, targets)
        assert math.isnan(value)

    # the defined half of the curve is still returned
    _, tpr, fpr = binary_auc(scores, torch.zeros(4))
    assert np.isnan(tpr).all()
    assert np.allclose(fpr, [0.0, 0.25, 0.5, 0.75, 1.0])
