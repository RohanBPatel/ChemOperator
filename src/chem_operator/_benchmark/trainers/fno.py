"""NeuralOperator FNO backend for the unified reactor benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from chem_operator._benchmark.trainers.base import (
    EpochReporter,
    TrainingOutcome,
    count_parameters,
    normalized_macro_rmse,
    seed_everything,
)


class FNOBackend:
    """Train and restore one- or two-dimensional FNO models."""

    name = "fno"

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.model_init: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.resume_state: Mapping[str, Any] | None = None

    def set_resume_state(self, state: Mapping[str, Any] | None) -> None:
        """Resume a Ray trial from a periodic model/optimizer checkpoint."""

        self.resume_state = state

    @staticmethod
    def _model(
        sample: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
        device: torch.device,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        from neuralop.models import FNO

        spatial_shape = tuple(int(value) for value in sample["x"].shape[1:])
        if len(spatial_shape) not in {1, 2}:
            raise ValueError("FNOBackend supports one- and two-dimensional grids.")
        requested_modes = int(config["modes"])
        modes = tuple(
            max(1, min(requested_modes, max(1, size // 2)))
            for size in spatial_shape
        )
        values = {
            "n_modes": modes,
            "in_channels": int(sample["x"].shape[0]),
            "out_channels": int(sample["y"].shape[0]),
            "hidden_channels": int(config["hidden_channels"]),
            "n_layers": int(config["n_layers"]),
            "positional_embedding": "grid",
        }
        return FNO(**values).to(device), values

    @staticmethod
    def _validation(
        model: torch.nn.Module,
        dataset: Dataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> float:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        squared: torch.Tensor | None = None
        count = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                target = batch["y"].to(device)
                prediction = model(batch["x"].to(device))
                reduction = tuple(range(2, target.ndim))
                batch_squared = (prediction - target).square().sum(
                    dim=(0,) + reduction
                )
                squared = (
                    batch_squared
                    if squared is None
                    else squared + batch_squared
                )
                count += target.shape[0]
                for size in target.shape[2:]:
                    count += 0  # dimensions are included below without copies
        if squared is None:
            raise ValueError("Validation dataset contains no samples.")
        points_per_sample = 1
        sample = dataset[0]["y"]
        for size in sample.shape[1:]:
            points_per_sample *= int(size)
        return float((squared / max(count * points_per_sample, 1)).sqrt().mean())

    def fit(
        self,
        train: Dataset,
        validation: Dataset,
        *,
        config: Mapping[str, Any],
        epochs: int,
        patience: int,
        seed: int,
        device: torch.device,
        reporter: EpochReporter | None = None,
    ) -> TrainingOutcome:
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must be positive.")
        seed_everything(seed)
        self.hyperparameters = dict(config)
        model, self.model_init = self._model(train[0], config, device)
        self.model = model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        start_epoch = 0
        if self.resume_state is not None:
            model.load_state_dict(self.resume_state["state_dict"], strict=True)
            optimizer_state = self.resume_state.get("optimizer_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = min(
                int(self.resume_state.get("epoch", 0)),
                epochs - 1,
            )
        generator = torch.Generator(
            device=torch.get_default_device()
        ).manual_seed(seed)
        loader = DataLoader(
            train,
            batch_size=min(int(config["batch_size"]), len(train)),
            shuffle=True,
            generator=generator,
        )
        history = {"train_loss": [], "valid_loss": []}
        best_loss = float("inf")
        best_epoch = 0
        best_state = deepcopy(model.state_dict())
        stale_epochs = 0
        for epoch in range(start_epoch + 1, epochs + 1):
            model.train()
            total = 0.0
            samples = 0
            for batch in loader:
                model_input = batch["x"].to(device)
                target = batch["y"].to(device)
                prediction = model(model_input)
                loss = normalized_macro_rmse(
                    prediction, target, channel_axis=1
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * target.shape[0]
                samples += target.shape[0]
            train_loss = total / max(samples, 1)
            valid_loss = self._validation(
                model,
                validation,
                batch_size=int(config["batch_size"]),
                device=device,
            )
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            improved = valid_loss < best_loss
            if improved:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            metrics: dict[str, float | int] = {
                "epoch": epoch,
                "train_normalized_rmse_macro": train_loss,
                "valid_normalized_rmse_macro": valid_loss,
                "best_valid_normalized_rmse_macro": best_loss,
                "n_params": count_parameters(model),
                "improved": int(improved),
            }
            if reporter is not None:
                reporter(
                    metrics,
                    model,
                    {
                        "epoch": epoch,
                        "optimizer_state": optimizer.state_dict(),
                        "model_init": self.model_init,
                        "hyperparameters": self.hyperparameters,
                    },
                )
            if stale_epochs >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        return TrainingOutcome(
            history=history,
            best_valid_loss=best_loss,
            best_epoch=best_epoch,
            parameter_count=count_parameters(model),
        )

    def predict(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Fit or load the FNO before prediction.")
        self.model.to(device).eval()
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in DataLoader(
                dataset, batch_size=max(1, batch_size), shuffle=False
            ):
                predictions.append(
                    self.model(batch["x"].to(device)).detach().cpu()
                )
        return torch.cat(predictions)

    def checkpoint_state(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No FNO model is available to checkpoint.")
        tensor_state = {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        return {
            "schema": "chem-operator-fno-v1",
            "backend": self.name,
            "model_init": self.model_init,
            "hyperparameters": self.hyperparameters,
            "state_dict": tensor_state,
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> None:
        from neuralop.models import FNO

        if state.get("backend") != self.name:
            raise ValueError("Checkpoint is not an FNO benchmark checkpoint.")
        self.model_init = dict(state["model_init"])
        if "n_modes" in self.model_init:
            self.model_init["n_modes"] = tuple(self.model_init["n_modes"])
        self.hyperparameters = dict(state.get("hyperparameters", {}))
        self.model = FNO(**self.model_init).to(device)
        incompatible = self.model.load_state_dict(
            state["state_dict"], strict=False
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "FNO checkpoint tensors do not match the architecture: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )
        self.model.eval()

    def predict_tensor(
        self,
        inputs: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("No FNO model is loaded.")
        self.model.to(device).eval()
        with torch.no_grad():
            return self.model(inputs.to(device)).detach().cpu()


__all__ = ["FNOBackend"]
