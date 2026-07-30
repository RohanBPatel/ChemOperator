"""Focused regression tests for benchmark-facing model adapters."""

from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

import chem_operator.models as model_module
from chem_operator.dataset_processing import (
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
)
from chem_operator.models import (
    CoordinateScaler,
    DeepONetBenchmarkConfig,
    DeepXDEAdapter,
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
    make_deeponet_dataloader,
    train_deeponet_lazy,
)


class _SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _one_dimensional_sample(offset: float) -> dict:
    coordinate = torch.linspace(0.0, 1.0, 5)
    field = torch.arange(5, dtype=torch.float32) + offset
    return {
        "input_fields": {"u": field[:1]},
        "output_fields": {"u": field[1:]},
        "constant_inputs": {"forcing": torch.tensor(2.0 + offset)},
        "input_coordinates": {"z": coordinate[:1]},
        "output_coordinates": {"z": coordinate[1:]},
        "metadata": {},
    }


def test_fno_adapter_supports_one_dimensional_profiles() -> None:
    raw = _SampleDataset(
        [_one_dimensional_sample(0.0), _one_dimensional_sample(1.0)]
    )
    inputs = (FNOChannel("forcing", "constant", "forcing"),)
    outputs = (FNOChannel("u", "field", "u"),)
    normalizer = fit_fno_zscore_normalizer(raw, inputs, outputs)
    adapter = FNOAdapter(
        raw,
        normalizer,
        input_channels=inputs,
        output_channels=outputs,
        coordinate_names=("z",),
    )

    item = adapter[0]
    assert item["x"].shape == (1, 5)
    assert item["y"].shape == (1, 5)
    assert item["z"].shape == (5,)
    torch.testing.assert_close(
        adapter.denormalize_output(item["y"]),
        adapter.physical_item(0)["y"],
    )
    assert adapter.checkpoint_config() == {
        "input_channels": [
            {
                "label": "forcing",
                "source": "constant",
                "key": "forcing",
                "species": None,
                "display_name": None,
                "unit": "-",
            }
        ],
        "output_channels": [
            {
                "label": "u",
                "source": "field",
                "key": "u",
                "species": None,
                "display_name": None,
                "unit": "-",
            }
        ],
        "coordinate_names": ["z"],
        "max_trajectories": None,
    }


def _two_dimensional_sample(offset: float) -> dict:
    z = torch.tensor([0.0, 0.5, 1.0])
    r = torch.tensor([0.0, 0.25])
    scalar = (
        torch.arange(z.numel() * r.numel(), dtype=torch.float32)
        .reshape(z.numel(), r.numel())
        .add(offset)
    )
    vector = torch.stack((scalar + 10.0, scalar + 20.0), dim=-1)
    return {
        "input_fields": {"u": scalar[:1], "v": vector[:1]},
        "output_fields": {"u": scalar[1:], "v": vector[1:]},
        "constant_inputs": {"forcing": torch.tensor(2.0 + offset)},
        "input_coordinates": {"z": z[:1], "r": r},
        "output_coordinates": {"z": z[1:], "r": r},
        "metadata": {},
    }


def _two_dimensional_adapter(
    samples: list[dict],
    *,
    format: str = "cartesian_product",
) -> DeepXDEAdapter:
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=("u", "v"),
            constant_field_order=("forcing",),
        ),
        normalization_config=NormalizationConfig(enabled=False),
    )
    return DeepXDEAdapter(
        _SampleDataset(samples),
        processor,
        format=format,
        coordinate_names=("z", "r"),
        include_constants=True,
    )


def test_deeponet_adapter_builds_a_two_dimensional_trunk_mesh() -> None:
    samples = [
        _two_dimensional_sample(0.0),
        _two_dimensional_sample(1.0),
    ]
    adapter = _two_dimensional_adapter(samples)
    item = adapter[0]

    expected_trunk = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.25],
            [0.5, 0.0],
            [0.5, 0.25],
            [1.0, 0.0],
            [1.0, 0.25],
        ]
    )
    scalar = torch.cat(
        (
            samples[0]["input_fields"]["u"],
            samples[0]["output_fields"]["u"],
        )
    )
    vector = torch.cat(
        (
            samples[0]["input_fields"]["v"],
            samples[0]["output_fields"]["v"],
        )
    )
    expected_target = torch.cat(
        (scalar.unsqueeze(-1), vector),
        dim=-1,
    ).reshape(-1, 3)

    assert item["branch"].shape == (7,)
    assert item["trunk"].shape == (6, 2)
    assert item["target"].shape == (6, 3)
    assert item["labels"] == ("u", "v[0]", "v[1]")
    torch.testing.assert_close(item["trunk"], expected_trunk)
    torch.testing.assert_close(item["target"], expected_target)

    arrays = adapter.arrays
    assert arrays.branch.shape == (2, 7)
    assert arrays.trunk.shape == (6, 2)
    assert arrays.targets.shape == (2, 6, 3)
    np.testing.assert_allclose(arrays.trunk, expected_trunk.numpy())

    scaler = CoordinateScaler.fit(arrays.trunk)
    loader = make_deeponet_dataloader(
        adapter,
        batch_size=2,
        shuffle=False,
        coordinate_scaler=scaler,
    )
    batch = next(iter(loader))
    assert batch["branch"].shape == (2, 7)
    assert batch["trunk"].shape == (6, 2)
    assert batch["target"].shape == (2, 6, 3)

    pointwise = _two_dimensional_adapter(samples, format="pointwise").arrays
    assert pointwise.branch.shape == (12, 7)
    assert pointwise.trunk.shape == (12, 2)
    assert pointwise.targets.shape == (12, 3)
    assert pointwise.trajectory_slices == (slice(0, 6), slice(6, 12))


class _TinyCartesianAdapter(Dataset):
    format = "cartesian_product"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        return {
            "branch": torch.zeros(2),
            "trunk": torch.tensor([[0.0], [1.0]]),
            "target": torch.zeros(2, 1),
            "coordinate": torch.tensor([0.0, 1.0]),
            "labels": ("u",),
        }


class _MarkerDeepONet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = torch.nn.Linear(2, 1, bias=False)
        self.trunk = torch.nn.Linear(1, 1, bias=False)
        self.marker = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        branch, trunk = inputs
        return branch[:, :1] + trunk[:, :1].T + self.marker


def test_lazy_deeponet_restores_the_best_validation_state(
    monkeypatch,
) -> None:
    fake_deepxde = SimpleNamespace(
        config=SimpleNamespace(set_random_seed=lambda seed: None)
    )
    monkeypatch.setitem(sys.modules, "deepxde", fake_deepxde)
    monkeypatch.setattr(
        model_module,
        "_network",
        lambda branch_width, trunk_width, output_width, config: (
            _MarkerDeepONet()
        ),
    )

    validation_losses = iter((0.1, 0.5, 0.4))
    training_epoch = 0

    def fake_loader_loss(
        model,
        loader,
        *,
        device,
        optimizer,
        pin_memory,
        loss_name,
    ):
        del loader, device, pin_memory, loss_name
        nonlocal training_epoch
        if optimizer is not None:
            training_epoch += 1
            with torch.no_grad():
                model.marker.fill_(float(training_epoch))
            return float(training_epoch)
        return next(validation_losses)

    monkeypatch.setattr(model_module, "_loader_loss", fake_loader_loss)
    callbacks: list[tuple[float, int, int]] = []

    def checkpoint_callback(model, metrics):
        callbacks.append(
            (
                float(model.marker.detach()),
                int(metrics["epoch"]),
                int(metrics["is_best"]),
            )
        )

    config = DeepONetBenchmarkConfig(
        epochs=3,
        batch_size=1,
        width=2,
        latent_width=1,
        display_every=10,
    )
    adapter = _TinyCartesianAdapter()
    model, history = train_deeponet_lazy(
        adapter,
        adapter,
        config=config,
        coordinate_scaler=CoordinateScaler.fit(
            np.asarray([[0.0], [1.0]], dtype=np.float32)
        ),
        checkpoint_callback=checkpoint_callback,
        device="cpu",
    )

    assert float(model.marker.detach()) == 1.0
    assert history.best_epoch == 1
    assert history.best_valid_loss == 0.1
    assert callbacks == [
        (1.0, 1, 1),
        (2.0, 2, 0),
        (3.0, 3, 0),
    ]
    assert DeepONetBenchmarkConfig.from_dict(config.to_dict()) == config
