"""Regression test: ``{mode}.best.pth`` must hold the best epoch's weights.

CheckpointCallback used to store ``obj=runner.model`` -- one live reference shared
by every ``_storage`` entry -- and re-save it under the "best" name at the end of
every epoch. The file therefore always held the *current* weights and became a
duplicate of ``{mode}.last.pth``. These tests pin the fixed behaviour.

Standalone by design: it drives a runner with a scripted validation loss so "best"
is unambiguous, and snapshots the weights itself to compare against.
"""

import copy
import os

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl

# validation loss per epoch; epoch 2 is best and training continues past it
SCRIPT = [0.90, 0.50, 0.70, 0.60]
BEST_EPOCH = 2


class _ScriptedRunner(dl.Runner):
    """Trains a tiny model and forces a known validation-loss trajectory."""

    def __init__(
        self, logdir, mode="model", topk=1, save_last=True, load_best_on_end=False
    ):
        super().__init__()
        self._logdir = logdir
        self._mode = mode
        self._topk = topk
        self._save_last = save_last
        self._load_best_on_end = load_best_on_end
        self.snapshots = {}  # epoch -> state_dict as of that epoch's end

    @property
    def num_epochs(self):
        return len(SCRIPT)

    def get_engine(self):
        return dl.CPUEngine()

    def get_loggers(self):
        return {}

    def get_model(self):
        torch.manual_seed(42)
        return torch.nn.Linear(4, 1)

    def get_criterion(self):
        return torch.nn.MSELoss()

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.5)

    def get_scheduler(self, optimizer):
        return None

    def get_loaders(self):
        torch.manual_seed(0)
        dataset = TensorDataset(torch.randn(16, 4), torch.randn(16, 1))
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_callbacks(self):
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                topk=self._topk,
                mode=self._mode,
                save_last=self._save_last,
                load_best_on_end=self._load_best_on_end,
            )
        }

    def handle_batch(self, batch):
        features, targets = batch
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics["loss"] = loss

    def on_loader_end(self, runner):
        if self.loader_key == "valid":
            self.loader_metrics["loss"] = SCRIPT[self.epoch_step - 1]
        else:
            self.loader_metrics["loss"] = float(self.batch_metrics["loss"].detach())
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        self.snapshots[self.epoch_step] = copy.deepcopy(self.model.state_dict())
        super().on_epoch_end(runner)


def _weights(path, mode):
    blob = torch.load(path, map_location="cpu")
    return blob["model_state_dict"] if mode == "runner" else blob


def _same(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[k], right[k]) for k in left
    )


@pytest.mark.parametrize(
    "mode,topk,save_last",
    [("model", 1, True), ("model", 3, True), ("model", 1, False), ("runner", 1, True)],
)
def test_best_checkpoint_holds_best_epoch(tmp_path, mode, topk, save_last):
    """``{mode}.best.pth`` holds the best epoch's weights, not the last ones."""
    runner = _ScriptedRunner(str(tmp_path), mode=mode, topk=topk, save_last=save_last)
    runner.run()

    best = _weights(os.path.join(str(tmp_path), f"{mode}.best.pth"), mode)
    assert _same(
        best, runner.snapshots[BEST_EPOCH]
    ), "best.pth is not the best epoch's weights"
    assert not _same(
        best, runner.snapshots[len(SCRIPT)]
    ), "best.pth drifted to the last epoch"

    kept = sorted(f for f in os.listdir(str(tmp_path)) if f.startswith(f"{mode}.0"))
    assert len(kept) == topk, f"expected {topk} per-epoch checkpoint(s), found {kept}"

    if save_last:
        last = _weights(os.path.join(str(tmp_path), f"{mode}.last.pth"), mode)
        assert _same(
            last, runner.snapshots[len(SCRIPT)]
        ), "last.pth is not the final weights"


def test_load_best_on_end_restores_best_epoch(tmp_path):
    """``load_best_on_end`` restores the best epoch's weights."""
    runner = _ScriptedRunner(str(tmp_path), load_best_on_end=True)
    runner.run()
    assert _same(runner.model.state_dict(), runner.snapshots[BEST_EPOCH])
