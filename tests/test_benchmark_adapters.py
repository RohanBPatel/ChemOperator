"""Shape and coordinate tests for common benchmark model adapters."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from chem_operator._benchmark.trainers.base import (
    GridDeepONetDataset,
    ResampledOperatorDataset,
)


class _OperatorDataset(Dataset):
    def __init__(self, spatial_shape: tuple[int, ...]):
        self.spatial_shape = spatial_shape

    def __len__(self):
        return 2

    def __getitem__(self, index):
        coordinates = {
            "z": torch.linspace(2.0, 4.0, self.spatial_shape[0])
        }
        if len(self.spatial_shape) == 2:
            coordinates["r"] = torch.linspace(
                0.0, 0.5, self.spatial_shape[1]
            )
        return {
            "x": torch.stack(
                [
                    torch.full(self.spatial_shape, float(index)),
                    torch.full(self.spatial_shape, 2.0),
                ]
            ),
            "y": torch.stack(
                [
                    torch.arange(
                        torch.tensor(self.spatial_shape).prod(),
                        dtype=torch.float32,
                    ).reshape(self.spatial_shape)
                ]
            ),
            **coordinates,
        }

    @staticmethod
    def denormalize_output(output):
        return output


def test_one_dimensional_resampling_and_deeponet_layout() -> None:
    common = ResampledOperatorDataset(
        _OperatorDataset((5,)),
        coordinate_names=("z",),
        output_labels=("T",),
        resample_shape=(8,),
    )
    deep = GridDeepONetDataset(
        common, coordinate_names=("z",), output_labels=("T",)
    )
    operator = common[0]
    sample = deep[0]
    assert operator["x"].shape == (2, 8)
    assert operator["y"].shape == (1, 8)
    assert sample["branch"].shape == (2,)
    assert sample["trunk"].shape == (8, 1)
    assert sample["target"].shape == (8, 1)
    torch.testing.assert_close(
        sample["trunk"][:, 0], torch.linspace(0.0, 1.0, 8)
    )


def test_two_dimensional_trunk_is_a_tensor_product_mesh() -> None:
    common = ResampledOperatorDataset(
        _OperatorDataset((3, 4)),
        coordinate_names=("z", "r"),
        output_labels=("velocity",),
    )
    deep = GridDeepONetDataset(
        common,
        coordinate_names=("z", "r"),
        output_labels=("velocity",),
    )
    sample = deep[1]
    assert sample["trunk"].shape == (12, 2)
    assert sample["target"].shape == (12, 1)
    assert sample["spatial_shape"] == (3, 4)
    assert torch.all((sample["trunk"] >= 0) & (sample["trunk"] <= 1))
