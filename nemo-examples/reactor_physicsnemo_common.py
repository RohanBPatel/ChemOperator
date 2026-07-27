"""Shared utilities for the reactor PhysicsNeMo examples.

This module provides:

- ``rel_l2``: global relative L2 error.
- ``save_comparison``: comparison and absolute-error plots.
- ``train_fno``: a small supervised 1D PhysicsNeMo FNO training routine.

The reactor scripts pass operator data in channel-last form, ``(N, L, C)``.
PhysicsNeMo's 1D FNO expects channel-first tensors, ``(N, C, L)``. The
``train_fno`` helper handles that conversion and returns a model that accepts
raw, channel-first inputs and returns denormalized, channel-first outputs.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    # Public import path in recent PhysicsNeMo versions.
    from physicsnemo.models.fno import FNO
except ImportError:  # pragma: no cover - compatibility with older releases
    from physicsnemo.models.fno.fno import FNO


__all__ = ["rel_l2", "save_comparison", "train_fno"]


def rel_l2(prediction: Tensor, target: Tensor, eps: float = 1.0e-12) -> float:
    """Return the global relative L2 error ``||prediction-target|| / ||target||``.

    Parameters
    ----------
    prediction, target:
        Tensors with identical shapes.
    eps:
        Minimum denominator used when the target norm is near zero.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    with torch.no_grad():
        error_norm = torch.linalg.vector_norm(prediction.detach() - target.detach())
        target_norm = torch.linalg.vector_norm(target.detach())
        denominator = target_norm.clamp_min(eps)
        return float((error_norm / denominator).cpu())


def _as_numpy(value: Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_field_matrix(
    value: Tensor | np.ndarray | Sequence[float],
    *,
    n_points: int,
    name: str,
) -> np.ndarray:
    """Convert a field to shape ``(n_points, n_channels)``."""
    array = np.squeeze(_as_numpy(value))

    if array.ndim == 1:
        if array.shape[0] != n_points:
            raise ValueError(
                f"{name} has {array.shape[0]} points, but coordinate has {n_points}"
            )
        return array[:, None]

    if array.ndim != 2:
        raise ValueError(
            f"{name} must reduce to one or two dimensions; got shape {array.shape}"
        )

    if array.shape[0] == n_points:
        return array
    if array.shape[1] == n_points:
        return array.T

    raise ValueError(
        f"{name} shape {array.shape} does not contain coordinate length {n_points}"
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "comparison"


def save_comparison(
    title: str,
    coordinate: Tensor | np.ndarray | Sequence[float],
    exact: Tensor | np.ndarray | Sequence[float],
    pinn: Tensor | np.ndarray | Sequence[float] | None,
    fno: Tensor | np.ndarray | Sequence[float] | None,
    deeponet: Tensor | np.ndarray | Sequence[float] | None,
    xlabel: str,
    ylabel: str,
    labels: Sequence[str] | None = None,
    *,
    output_dir: str | Path = "Figures",
    dpi: int = 180,
) -> Path:
    """Save solution-comparison and absolute-error plots.

    The positional signature matches the CSTR, PFR, and packed-bed scripts.
    ``pinn``, ``fno``, or ``deeponet`` may be ``None``. For multiple output
    channels, one row is created per channel.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.
    """
    x = np.ravel(_as_numpy(coordinate))
    if x.ndim != 1:
        raise ValueError("coordinate must be one-dimensional")

    fields: dict[str, np.ndarray] = {
        "Exact": _as_field_matrix(exact, n_points=len(x), name="exact")
    }
    if pinn is not None:
        fields["PINN"] = _as_field_matrix(pinn, n_points=len(x), name="pinn")
    if fno is not None:
        fields["FNO"] = _as_field_matrix(fno, n_points=len(x), name="fno")
    if deeponet is not None:
        fields["DeepONet"] = _as_field_matrix(
            deeponet, n_points=len(x), name="deeponet"
        )

    n_channels = fields["Exact"].shape[1]
    for method, values in fields.items():
        if values.shape[1] != n_channels:
            raise ValueError(
                f"{method} has {values.shape[1]} channels; expected {n_channels}"
            )

    if labels is None:
        channel_labels = [ylabel] if n_channels == 1 else [
            f"{ylabel} {index + 1}" for index in range(n_channels)
        ]
    else:
        channel_labels = list(labels)
        if len(channel_labels) != n_channels:
            raise ValueError(
                f"labels contains {len(channel_labels)} entries; expected {n_channels}"
            )

    # Each output channel gets a solution panel and an absolute-error panel.
    figure, axes = plt.subplots(
        n_channels,
        2,
        figsize=(11.0, max(3.8, 3.5 * n_channels)),
        squeeze=False,
        constrained_layout=True,
    )

    truth = fields["Exact"]
    for channel in range(n_channels):
        solution_axis = axes[channel, 0]
        error_axis = axes[channel, 1]

        solution_axis.plot(x, truth[:, channel], label="Exact", linewidth=2.2)
        for method, values in fields.items():
            if method == "Exact":
                continue
            solution_axis.plot(x, values[:, channel], label=method, linewidth=1.6)
            error_axis.plot(
                x,
                np.abs(values[:, channel] - truth[:, channel]),
                label=method,
                linewidth=1.6,
            )

        solution_axis.set_title(channel_labels[channel])
        solution_axis.set_xlabel(xlabel)
        solution_axis.set_ylabel(channel_labels[channel])
        solution_axis.grid(True, alpha=0.25)
        solution_axis.legend()

        error_axis.set_title(f"Absolute error: {channel_labels[channel]}")
        error_axis.set_xlabel(xlabel)
        error_axis.set_ylabel("absolute error")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend()

    figure.suptitle(title)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slugify(title)}_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


class _NormalizedFNO(nn.Module):
    """Apply channelwise normalization around a PhysicsNeMo FNO."""

    def __init__(
        self,
        model: nn.Module,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "The returned 1D FNO expects input shape (batch, channels, points); "
                f"got {tuple(x.shape)}"
            )
        normalized = (x - self.input_mean) / self.input_std
        prediction = self.model(normalized)
        return prediction * self.output_std + self.output_mean


def _channel_statistics(values: Tensor, eps: float = 1.0e-6) -> tuple[Tensor, Tensor]:
    """Compute statistics for channel-first ``(N, C, L)`` tensors."""
    mean = values.mean(dim=(0, 2), keepdim=True)
    std = values.std(dim=(0, 2), keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def _relative_l2_batch(prediction: Tensor, target: Tensor, eps: float = 1.0e-12) -> Tensor:
    error = torch.linalg.vector_norm(
        (prediction - target).flatten(start_dim=1), dim=1
    )
    scale = torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1).clamp_min(eps)
    return (error / scale).mean()


def train_fno(
    x: Tensor,
    y: Tensor,
    *,
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-6,
    validation_fraction: float = 0.15,
    latent_channels: int = 32,
    num_fno_layers: int = 4,
    num_fno_modes: int = 12,
    padding: int = 4,
    decoder_layers: int = 2,
    decoder_layer_size: int = 64,
    patience: int = 150,
    min_delta: float = 1.0e-6,
    seed: int = 17,
    verbose: bool = True,
) -> nn.Module:
    """Train and return a normalized 1D PhysicsNeMo Fourier neural operator.

    Parameters
    ----------
    x, y:
        Channel-last tensors with shapes ``(n_samples, n_points, n_inputs)`` and
        ``(n_samples, n_points, n_outputs)``. This is the layout produced by the
        three reactor ``operator_data`` functions.

    Notes
    -----
    The returned model accepts channel-first raw inputs of shape
    ``(batch, n_inputs, n_points)`` and returns raw, denormalized predictions of
    shape ``(batch, n_outputs, n_points)``. This exactly matches the calls in the
    reactor scripts.
    """
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("x and y must be torch.Tensor objects")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(
            "x and y must have shapes (samples, points, channels); "
            f"got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    if x.shape[:2] != y.shape[:2]:
        raise ValueError(
            "x and y must have the same sample and point dimensions; "
            f"got {tuple(x.shape[:2])} and {tuple(y.shape[:2])}"
        )
    if x.shape[0] < 2:
        raise ValueError("At least two operator samples are required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    device = x.device
    if y.device != device:
        y = y.to(device)

    # FNO expects (N, C, L). Convert once and keep training batches on-device.
    x_cf = x.detach().to(dtype=torch.float32).permute(0, 2, 1).contiguous()
    y_cf = y.detach().to(device=device, dtype=torch.float32).permute(0, 2, 1).contiguous()

    n_samples, in_channels, n_points = x_cf.shape
    out_channels = y_cf.shape[1]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator).to(device)

    if validation_fraction == 0.0:
        n_validation = 0
    else:
        n_validation = max(1, round(validation_fraction * n_samples))
        n_validation = min(n_validation, n_samples - 1)

    validation_indices = permutation[:n_validation]
    training_indices = permutation[n_validation:]

    x_train = x_cf.index_select(0, training_indices)
    y_train = y_cf.index_select(0, training_indices)
    x_validation = (
        x_cf.index_select(0, validation_indices) if n_validation else None
    )
    y_validation = (
        y_cf.index_select(0, validation_indices) if n_validation else None
    )

    input_mean, input_std = _channel_statistics(x_train)
    output_mean, output_std = _channel_statistics(y_train)

    # A real FFT has at most floor(L/2)+1 distinct frequency bins.
    available_modes = max(1, n_points // 2)
    modes = max(1, min(int(num_fno_modes), available_modes))

    core = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=1,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        num_fno_modes=modes,
        padding=padding,
        # The reactor scripts already supply the coordinate as an input channel.
        coord_features=False,
    ).to(device)

    model = _NormalizedFNO(
        core,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metric = math.inf
    stale_epochs = 0
    n_train = training_indices.numel()
    batch_size = min(batch_size, n_train)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_order = training_indices[
            torch.randperm(n_train, device=device)
        ]

        running_loss = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            batch_indices = epoch_order[start : start + batch_size]
            xb = x_cf.index_select(0, batch_indices)
            yb = y_cf.index_select(0, batch_indices)

            prediction = model(xb)
            # Channel normalization is already built into the model, but using the
            # normalized residual here prevents a high-magnitude output channel from
            # dominating a multi-output problem.
            loss = F.mse_loss(
                (prediction - yb) / output_std,
                torch.zeros_like(yb),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = xb.shape[0]
            running_loss += float(loss.detach()) * count
            seen += count

        scheduler.step()

        model.eval()
        with torch.no_grad():
            if x_validation is not None and y_validation is not None:
                metric = float(_relative_l2_batch(model(x_validation), y_validation))
            else:
                metric = float(_relative_l2_batch(model(x_train), y_train))

        improved = metric < best_metric - min_delta
        if improved:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        should_report = epoch == 1 or epoch % 100 == 0 or epoch == epochs
        if verbose and should_report:
            average_loss = running_loss / max(1, seen)
            split_name = "validation" if n_validation else "training"
            print(
                f"FNO epoch {epoch:4d}/{epochs}: "
                f"normalized MSE={average_loss:.3e}, "
                f"{split_name} relative L2={metric:.3e}"
            )

        if patience > 0 and stale_epochs >= patience:
            if verbose:
                print(
                    f"FNO early stopping at epoch {epoch}; "
                    f"best relative L2={best_metric:.3e}"
                )
            break

    model.load_state_dict(best_state)
    model.eval()
    return model
