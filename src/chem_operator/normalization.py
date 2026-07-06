from typing import Protocol, Literal
import torch

class Normalizer(Protocol):
    def normalize(
        self,
        x: torch.Tensor,
        field: str,
    ) -> torch.Tensor:
        ...

    def denormalize(
        self,
        x: torch.Tensor,
        field: str,
    ) -> torch.Tensor:
        ...

    def delta_normalize(
        self,
        x: torch.Tensor,
        field: str,
    ) -> torch.Tensor:
        ...

    def delta_denormalize(
        self,
        x: torch.Tensor,
        field: str,
    ) -> torch.Tensor:
        ...

    def normalize_flattened(
        self,
        x: torch.Tensor,
        mode: Literal["variable", "constant"],
    ) -> torch.Tensor:
        ...

    def denormalize_flattened(
        self,
        x: torch.Tensor,
        mode: Literal["variable", "constant"],
    ) -> torch.Tensor:
        ...

    def delta_normalize_flattened(
        self,
        x: torch.Tensor,
        mode: Literal["variable"],
    ) -> torch.Tensor:
        ...

    def delta_denormalize_flattened(
        self,
        x: torch.Tensor,
        mode: Literal["variable"],
    ) -> torch.Tensor:
        ...

# ZScoreNormalizer
# RMSNormalizer
# MinMaxNormalizer
# IdentityNormalizer

# old
# class Standardizer:
#     def fit(self, x):
#         self.mean = x.mean(axis=0, keepdims=True)
#         self.std = x.std(axis=0, keepdims=True)
#         self.std = np.maximum(self.std, 1e-8)
#         return self

#     def transform(self, x):
#         return ((x - self.mean) / self.std).astype(np.float32)

#     def inverse_transform(self, x):
#         return x * self.std + self.mean
