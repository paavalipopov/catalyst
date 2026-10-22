"""Checkpoint ownership across two simulated nodes, each with two ranks."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from accelerate.state import DistributedType
import torch

from catalyst import dl
from catalyst.callbacks import checkpoint as checkpoint_module


def _checkpoint_object(kind, rank):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(rank)
        model.bias.fill_(rank)
    if kind == "dict":
        return {"net": model}
    if kind == "runner":
        return {"model": model, "epoch_step": 1}
    return model


def _engine(rank, writes, distributed_type):
    def save(checkpoint, path):
        # Match older Accelerate: each node's local rank 0 writes.
        if rank % 2 == 0:
            writes.append(rank)
            torch.save(checkpoint, path)

    return SimpleNamespace(
        distributed_type=distributed_type,
        is_main_process=rank == 0,
        wait_for_everyone=Mock(),
        unwrap_model=lambda model: model,
        save=save,
    )


@pytest.mark.parametrize("kind", ["model", "dict", "runner"])
def test_checkpoint_writes_only_global_main(tmp_path, kind):
    """A second node's local main process must not overwrite shared checkpoints."""
    mode = "runner" if kind == "runner" else "model"
    callback = dl.CheckpointCallback(str(tmp_path), mode=mode)
    writes = []
    for rank in range(4):
        engine = _engine(rank, writes, DistributedType.MULTI_CPU)
        path = callback._save(
            SimpleNamespace(engine=engine),
            _checkpoint_object(kind, rank),
            str(tmp_path / mode),
        )
        # Non-writing ranks still join the checkpoint synchronization.
        engine.wait_for_everyone.assert_called()

    assert writes == [0]
    checkpoint = torch.load(path, map_location="cpu")
    if kind == "dict":
        checkpoint = checkpoint["model_net_state_dict"]
    elif kind == "runner":
        checkpoint = checkpoint["model_state_dict"]
    assert torch.equal(checkpoint["weight"], torch.zeros(1, 1))


@pytest.mark.parametrize("kind", ["model", "dict", "runner"])
def test_xla_checkpoint_save_remains_collective(tmp_path, monkeypatch, kind):
    """Every XLA rank joins saving, with global rather than per-node ownership."""
    mode = "runner" if kind == "runner" else "model"
    callback = dl.CheckpointCallback(str(tmp_path), mode=mode)
    participants, writes = [], []
    for rank in range(4):

        def save(checkpoint, path, master_only=True, global_master=False):
            participants.append(rank)
            if not master_only or (rank == 0 if global_master else rank % 2 == 0):
                writes.append(rank)
                torch.save(checkpoint, path)

        monkeypatch.setattr(
            checkpoint_module, "xm", SimpleNamespace(save=save), raising=False
        )
        engine = _engine(rank, writes, "XLA")  # TPU before accelerate 0.28
        callback._save(
            SimpleNamespace(engine=engine),
            _checkpoint_object(kind, rank),
            str(tmp_path / mode),
        )
        engine.wait_for_everyone.assert_called()

    assert participants == [0, 1, 2, 3]
    assert writes == [0]
