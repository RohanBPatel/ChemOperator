"""Direct and POD DeepONet backends for common reactor operator grids."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chem_operator._benchmark.trainers.base import (
    EpochReporter,
    GridDeepONetDataset,
    TrainingOutcome,
)
from chem_operator.models import (
    CoordinateScaler,
    DeepONetBenchmarkConfig,
    PODTransform,
    _PODOnlyDeepONet,
    _network,
    _with_output_channel,
    deeponet_parameter_counts,
    fit_incremental_pod_dataset,
    make_deeponet_dataloader,
    train_deeponet_lazy,
)
from chem_operator.utils import to_numpy


class DeepONetBackend:
    """DeepXDE-backed direct or POD-only DeepONet model family."""

    def __init__(self, *, use_pod: bool = False) -> None:
        self.use_pod = use_pod
        self.name = "pod-deeponet" if use_pod else "deeponet"
        self.model: torch.nn.Module | None = None
        self.pod: PODTransform | None = None
        self.coordinate_scaler: CoordinateScaler | None = None
        self.model_init: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.resume_state: Mapping[str, Any] | None = None

    def set_resume_state(self, state: Mapping[str, Any] | None) -> None:
        """Resume a Ray trial from saved model weights."""

        self.resume_state = state

    @staticmethod
    def _benchmark_config(
        values: Mapping[str, Any],
        *,
        epochs: int,
        seed: int,
    ) -> DeepONetBenchmarkConfig:
        return DeepONetBenchmarkConfig(
            loss=str(values.get("loss", "mse")),
            epochs=epochs,
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values.get("weight_decay", 0.0)),
            batch_size=int(values["batch_size"]),
            width=int(values["width"]),
            latent_width=int(values.get("latent_width", 16)),
            branch_hidden_layers=int(values.get("branch_hidden_layers", 2)),
            trunk_hidden_layers=int(values.get("trunk_hidden_layers", 2)),
            activation=str(values.get("activation", "gelu")),
            display_every=max(1, min(epochs, 10)),
            variance_threshold=float(values.get("variance_threshold", 0.999)),
            seed=seed,
        )

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
        if not isinstance(train, GridDeepONetDataset) or not isinstance(
            validation, GridDeepONetDataset
        ):
            raise TypeError("DeepONetBackend requires GridDeepONetDataset inputs.")
        self.hyperparameters = dict(config)
        network_config = self._benchmark_config(
            config, epochs=epochs, seed=seed
        )
        first = train[0]
        self.coordinate_scaler = CoordinateScaler.fit(
            to_numpy(first["trunk"])
        )
        if self.use_pod:
            self.pod = fit_incremental_pod_dataset(
                train,
                variance_threshold=float(
                    config.get("variance_threshold", 0.999)
                ),
                max_components=int(config.get("max_pod_components", 64)),
            )
        else:
            self.pod = None
        self.model_init = {
            "branch_width": int(first["branch"].shape[-1]),
            "trunk_width": int(first["trunk"].shape[-1]),
            "output_width": int(first["target"].shape[-1]),
            "network_config": network_config.to_dict(),
            "use_pod": self.use_pod,
        }
        best_macro = float("inf")
        best_macro_epoch = 0
        best_macro_state: dict[str, torch.Tensor] | None = None

        def checkpoint_callback(
            model: torch.nn.Module,
            raw_metrics: Mapping[str, float | int],
        ) -> None:
            nonlocal best_macro, best_macro_epoch, best_macro_state
            valid_loss = self._validation_macro(
                model,
                validation,
                batch_size=network_config.batch_size,
                device=device,
            )
            improved = valid_loss < best_macro
            if improved:
                best_macro = valid_loss
                best_macro_epoch = int(raw_metrics["epoch"])
                best_macro_state = deepcopy(model.state_dict())
            if reporter is None:
                return
            metrics = {
                **raw_metrics,
                "valid_normalized_rmse_macro": valid_loss,
                "best_valid_normalized_rmse_macro": best_macro,
                "improved": int(improved),
            }
            reporter(
                metrics,
                model,
                {
                    "epoch": int(raw_metrics["epoch"]),
                    "model_init": self.model_init,
                    "hyperparameters": self.hyperparameters,
                },
            )

        self.model, history = train_deeponet_lazy(
            train,
            validation,
            config=network_config,
            coordinate_scaler=self.coordinate_scaler,
            pod=self.pod,
            checkpoint_callback=checkpoint_callback,
            early_stopping_patience=patience,
            initial_state_dict=(
                None
                if self.resume_state is None
                else self.resume_state.get("state_dict")
            ),
            start_epoch=(
                0
                if self.resume_state is None
                else min(
                    int(self.resume_state.get("epoch", 0)),
                    epochs - 1,
                )
            ),
            device=device,
        )
        if best_macro_state is None:
            raise RuntimeError("DeepONet did not produce a validation checkpoint.")
        self.model.load_state_dict(best_macro_state)
        self.model.eval()
        counts = deeponet_parameter_counts(self.model)
        return TrainingOutcome(
            history={
                "train_loss": [
                    float(values[0]) for values in history.loss_train
                ],
                "valid_loss": [
                    float(values[0]) for values in history.loss_test
                ],
            },
            best_valid_loss=best_macro,
            best_epoch=best_macro_epoch,
            parameter_count=int(counts["n_params"]),
            extra={
                "n_params_branch": int(counts["n_params_branch"]),
                "n_params_trunk": int(counts["n_params_trunk"]),
                "pod_components": (
                    0 if self.pod is None else self.pod.n_components
                ),
            },
        )

    def _validation_macro(
        self,
        model: torch.nn.Module,
        dataset: GridDeepONetDataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> float:
        if self.coordinate_scaler is None:
            raise RuntimeError("DeepONet coordinate scaling is unavailable.")
        output_width = int(dataset[0]["target"].shape[-1])
        squared = torch.zeros(
            output_width,
            dtype=torch.float64,
            device="cpu",
        )
        count = 0
        model.eval()
        with torch.no_grad():
            loader = make_deeponet_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                coordinate_scaler=self.coordinate_scaler,
            )
            for batch in loader:
                target = batch["target"].to(device)
                prediction = model(
                    (
                        batch["branch"].to(device),
                        batch["trunk"].to(device),
                    )
                )
                if self.pod is not None:
                    prediction = self.pod.unflatten_tensor(prediction)
                else:
                    prediction = _with_output_channel(
                        prediction, output_width
                    )
                squared += (
                    (prediction - target)
                    .square()
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .to(torch.float64)
                )
                count += int(target.shape[0] * target.shape[1])
        return float((squared / max(count, 1)).sqrt().mean())

    def _loader(
        self,
        dataset: GridDeepONetDataset,
        *,
        batch_size: int,
    ):
        if self.coordinate_scaler is None:
            raise RuntimeError("DeepONet coordinate scaling is unavailable.")
        return make_deeponet_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            coordinate_scaler=self.coordinate_scaler,
        )

    def predict(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Fit or load the DeepONet before prediction.")
        if not isinstance(dataset, GridDeepONetDataset):
            raise TypeError("DeepONet prediction needs a GridDeepONetDataset.")
        self.model.to(device).eval()
        point_predictions: list[torch.Tensor] = []
        output_width = int(self.model_init["output_width"])
        with torch.no_grad():
            for batch in self._loader(dataset, batch_size=batch_size):
                prediction = self.model(
                    (
                        batch["branch"].to(device),
                        batch["trunk"].to(device),
                    )
                )
                if self.pod is not None:
                    prediction = self.pod.unflatten_tensor(prediction)
                else:
                    prediction = _with_output_channel(
                        prediction, output_width
                    )
                point_predictions.append(prediction.detach().cpu())
        values = torch.cat(point_predictions)
        spatial_shape = dataset.spatial_shape
        values = values.reshape(
            values.shape[0], *spatial_shape, output_width
        )
        return values.movedim(-1, 1)

    @staticmethod
    def _pod_state(pod: PODTransform | None) -> dict[str, Any] | None:
        if pod is None:
            return None
        return {
            "mean": torch.from_numpy(np.asarray(pod.mean)).cpu(),
            "basis": torch.from_numpy(np.asarray(pod.basis)).cpu(),
            "explained_variance_ratio": torch.from_numpy(
                np.asarray(pod.explained_variance_ratio)
            ).cpu(),
            "cumulative_explained_variance": float(
                pod.cumulative_explained_variance
            ),
            "output_shape": list(pod.output_shape),
        }

    @staticmethod
    def _restore_pod(values: Mapping[str, Any] | None) -> PODTransform | None:
        if values is None:
            return None

        def array(name: str) -> np.ndarray:
            return to_numpy(values[name])

        return PODTransform(
            mean=array("mean"),
            basis=array("basis"),
            explained_variance_ratio=array("explained_variance_ratio"),
            cumulative_explained_variance=float(
                values["cumulative_explained_variance"]
            ),
            output_shape=tuple(int(v) for v in values["output_shape"]),
        )

    def checkpoint_state(self) -> dict[str, Any]:
        if self.model is None or self.coordinate_scaler is None:
            raise RuntimeError("No DeepONet is available to checkpoint.")
        return {
            "schema": "chem-operator-deeponet-v1",
            "backend": self.name,
            "model_init": self.model_init,
            "hyperparameters": self.hyperparameters,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "coordinate_scaler": {
                "minimum": torch.from_numpy(
                    np.asarray(self.coordinate_scaler.minimum)
                ),
                "span": torch.from_numpy(
                    np.asarray(self.coordinate_scaler.span)
                ),
            },
            "pod": self._pod_state(self.pod),
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> None:
        if state.get("backend") != self.name:
            raise ValueError(
                f"Checkpoint backend {state.get('backend')!r} does not match "
                f"{self.name!r}."
            )
        self.model_init = dict(state["model_init"])
        self.hyperparameters = dict(state.get("hyperparameters", {}))
        scaler = state["coordinate_scaler"]
        self.coordinate_scaler = CoordinateScaler(
            minimum=to_numpy(scaler["minimum"]),
            span=to_numpy(scaler["span"]),
        )
        self.pod = self._restore_pod(state.get("pod"))
        config = DeepONetBenchmarkConfig.from_dict(
            self.model_init["network_config"]
        )
        if self.use_pod:
            if self.pod is None:
                raise ValueError("POD-DeepONet checkpoint has no POD state.")
            self.model = _PODOnlyDeepONet(
                int(self.model_init["branch_width"]),
                self.pod,
                config,
            )
        else:
            self.model = _network(
                int(self.model_init["branch_width"]),
                int(self.model_init["trunk_width"]),
                int(self.model_init["output_width"]),
                config,
            )
        self.model.to(device)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.eval()

    def predict_tensor(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("No DeepONet is loaded.")
        branch, trunk = inputs
        self.model.to(device).eval()
        with torch.no_grad():
            prediction = self.model(
                (branch.to(device), trunk.to(device))
            )
            if self.pod is not None:
                prediction = self.pod.unflatten_tensor(prediction)
            else:
                prediction = _with_output_channel(
                    prediction, int(self.model_init["output_width"])
                )
        return prediction.detach().cpu()


__all__ = ["DeepONetBackend"]
