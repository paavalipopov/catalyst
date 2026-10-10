# flake8: noqa
"""2-process CPU (gloo) DDP checks for loader-metric reduction and checkpointing.

Every rank reports its own validation loss, so the tests can tell a reduced metric
from a per-rank one: rank 0 alone would pick epoch 2 as best, rank 1 alone epoch 3,
and their mean picks epoch 3.
"""
from datetime import timedelta
import json
import os
import socket

import pytest

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl

SCRIPT = {0: [0.90, 0.50, 0.70, 0.60], 1: [0.90, 0.80, 0.30, 0.60]}
MEAN_SCRIPT = [0.90, 0.65, 0.50, 0.60]
BEST_EPOCH = 3


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _DDPRunner(dl.Runner):
    """Trains a tiny model on two CPU ranks with a scripted per-rank valid loss."""

    def __init__(self, logdir, mode, port):
        super().__init__()
        self._logdir, self._mode, self._port = logdir, mode, port

    @property
    def num_epochs(self):
        return len(MEAN_SCRIPT)

    def get_engine(self):
        return dl.DistributedDataParallelEngine(
            port=self._port,
            num_node_workers=2,
            process_group_kwargs={"backend": "gloo", "timeout": timedelta(seconds=120)},
            cpu=True,
        )

    def get_loggers(self):
        return {}

    def get_model(self):
        torch.manual_seed(42)
        return torch.nn.Linear(4, 1)

    def get_criterion(self):
        return torch.nn.MSELoss()

    def get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def get_scheduler(self, optimizer):
        return None

    def get_loaders(self):
        generator = torch.Generator().manual_seed(0)
        features = torch.randn(32, 4, generator=generator)
        targets = torch.randn(32, 1, generator=generator)
        loader = DataLoader(TensorDataset(features, targets), batch_size=4)
        return {"train": loader, "valid": loader}

    def get_callbacks(self):
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                topk=1,
                mode=self._mode,
                load_best_on_end=self._mode == "model",
            )
        }

    def handle_batch(self, batch):
        features, targets = batch
        loss = self.criterion(self.model(features), targets)
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.batch_metrics["loss"] = loss.detach()

    def on_loader_end(self, runner):
        rank = self.engine.process_index
        if self.loader_key == "valid":
            self.loader_metrics["loss"] = SCRIPT[rank][self.epoch_step - 1]
        else:
            self.loader_metrics["loss"] = float(self.batch_metrics["loss"])
        self.loader_metrics["rank"] = float(rank)
        super().on_loader_end(runner)

    def on_epoch_end(self, runner):
        if self.engine.is_main_process:
            state_dict = self.engine.unwrap_model(self.model).state_dict()
            path = os.path.join(self._logdir, f"snapshot.{self.epoch_step}.pth")
            torch.save(state_dict, path)
        super().on_epoch_end(runner)

    def on_experiment_end(self, runner):
        state_dict = self.engine.unwrap_model(self.model).state_dict()
        report = {
            "valid": [
                self.experiment_metrics[e]["valid"]
                for e in sorted(self.experiment_metrics)
            ],
            "storage": [
                os.path.basename(x.logpath)
                for x in self.callbacks["checkpoint"]._storage
            ],
            "final": {k: v.tolist() for k, v in state_dict.items()},
        }
        path = os.path.join(self._logdir, f"rank{self.engine.process_index}.json")
        with open(path, "w") as fout:
            json.dump(report, fout)
        super().on_experiment_end(runner)


def _run(logdir, mode):
    runner = _DDPRunner(str(logdir), mode=mode, port=_free_port())
    runner.run()
    reports = []
    for rank in range(2):
        with open(os.path.join(str(logdir), f"rank{rank}.json")) as fin:
            reports.append(json.load(fin))
    return reports


def _load_weights(path, mode):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state_dict"] if mode == "runner" else checkpoint


def _same(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[k], right[k]) for k in left
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.parametrize("mode", ["model", "runner"])
def test_ddp_metrics_and_best_checkpoint(tmp_path, mode):
    reports = _run(tmp_path, mode)

    # loader metrics are averaged over ranks, so both ranks log the same values
    for report in reports:
        assert [m["rank"] for m in report["valid"]] == [0.5] * len(MEAN_SCRIPT)
        assert [m["loss"] for m in report["valid"]] == pytest.approx(MEAN_SCRIPT)
        assert report["storage"] == [f"{mode}.{BEST_EPOCH:04d}.pth"]

    # best.pth holds the epoch that is best by the averaged metric
    snapshots = {
        e: torch.load(str(tmp_path / f"snapshot.{e}.pth"))
        for e in range(1, len(MEAN_SCRIPT) + 1)
    }
    best = _load_weights(str(tmp_path / f"{mode}.best.pth"), mode)
    last = _load_weights(str(tmp_path / f"{mode}.last.pth"), mode)
    assert _same(best, snapshots[BEST_EPOCH])
    assert _same(last, snapshots[len(MEAN_SCRIPT)])
    assert sorted(p.name for p in tmp_path.glob(f"{mode}.0*.pth")) == [
        f"{mode}.{BEST_EPOCH:04d}.pth"
    ]

    if mode == "model":
        # load_best_on_end restores the same weights on every rank
        for report in reports:
            final = {k: torch.tensor(v) for k, v in report["final"].items()}
            assert _same(final, snapshots[BEST_EPOCH])
