"""External DeepONet time-stepper for the Cantera hydrogen/oxygen PFR example.

The DeepONet learns the finite-time chemistry map

    (T(t), Y(t), dt) -> Y(t + dt) - Y(t)

from trajectories generated with Cantera's standard ReactorNet solver. During
rollout, temperature is recovered from constant enthalpy and pressure, which
matches the adiabatic constant-pressure Lagrangian PFR formulation.

Requires: cantera >= 3.2, torch, numpy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# PFR configuration
# -----------------------------------------------------------------------------
REACTION_MECHANISM = "h2o2.yaml"
PRESSURE = ct.one_atm
LENGTH = 1.5e-7
U0 = 0.006
AREA = 1.0e-4
T_TOTAL = LENGTH / U0
BASE_COMPOSITION = "H2:2, O2:1, AR:0.1"


def set_inlet(gas: ct.Solution, temperature: float, phi: float) -> None:
    """Set a hydrogen/oxygen/argon inlet state."""
    gas.TP = temperature, PRESSURE
    gas.set_equivalence_ratio(phi, fuel="H2", oxidizer="O2:1, AR:0.1")


def solve_lagrangian(
    temperature: float,
    phi: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Reference PFR trajectory using Cantera's traditional time integrator."""
    gas = ct.Solution(REACTION_MECHANISM)
    set_inlet(gas, temperature, phi)

    mass_flow_rate = U0 * gas.density * AREA
    reactor = ct.IdealGasConstPressureReactor(
        gas, energy="on", clone=True
    )
    network = ct.ReactorNet([reactor])

    dt = T_TOTAL / n_steps
    times = np.arange(n_steps + 1, dtype=float) * dt
    z = np.zeros(n_steps + 1)
    states = np.empty((n_steps + 1, gas.n_species + 1))
    states[0] = np.r_[reactor.phase.T, reactor.phase.Y]

    for i in range(1, n_steps + 1):
        network.advance(times[i])
        velocity = mass_flow_rate / (AREA * reactor.phase.density)
        z[i] = z[i - 1] + velocity * dt
        states[i] = np.r_[reactor.phase.T, reactor.phase.Y]

    return {"t": times, "z": z, "state": states}


def solve_reactor_chain(
    temperature: float,
    phi: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Second traditional PFR approximation: a chain of stirred reactors."""
    gas = ct.Solution(REACTION_MECHANISM)
    set_inlet(gas, temperature, phi)

    mass_flow_rate = U0 * gas.density * AREA
    dz = LENGTH / n_steps

    reactor = ct.IdealGasReactor(gas, energy="on", clone=True)
    reactor.volume = AREA * dz
    upstream = ct.Reservoir(gas, clone=True)
    downstream = ct.Reservoir(gas, clone=True)
    mfc = ct.MassFlowController(upstream, reactor, mdot=mass_flow_rate)
    _ = ct.PressureController(reactor, downstream, primary=mfc, K=1e-12)

    network = ct.ReactorNet([reactor])
    network.max_time_step = 1e4

    z = (np.arange(n_steps) + 1) * dz
    times = np.zeros(n_steps)
    states = np.empty((n_steps, gas.n_species + 1))

    for i in range(n_steps):
        upstream.phase.TDY = reactor.phase.TDY
        network.reinitialize()
        network.solve_steady()
        times[i] = (times[i - 1] if i else 0.0) + reactor.mass / mass_flow_rate
        states[i] = np.r_[reactor.phase.T, reactor.phase.Y]

    return {"t": times, "z": z, "state": states}


# -----------------------------------------------------------------------------
# DeepONet
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for n_in, n_out in zip(sizes[:-2], sizes[1:-1]):
            layers.extend((nn.Linear(n_in, n_out), nn.SiLU()))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONetStepper(nn.Module):
    """Branch: current state. Trunk: requested time increment."""

    def __init__(self, state_dim: int, output_dim: int, width: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.width = width
        self.branch = MLP([state_dim, 128, 128, output_dim * width])
        self.trunk = MLP([1, 64, 64, width])
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, state: torch.Tensor, log_dt: torch.Tensor) -> torch.Tensor:
        branch = self.branch(state).view(-1, self.output_dim, self.width)
        trunk = self.trunk(log_dt)
        return torch.einsum("bow,bw->bo", branch, trunk) + self.bias


@dataclass
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    log_dt_mean: float
    log_dt_std: float

    def encode_state(self, x: np.ndarray) -> np.ndarray:
        return (x - self.state_mean) / self.state_std

    def encode_dt(self, dt: np.ndarray) -> np.ndarray:
        return (np.log10(dt) - self.log_dt_mean) / self.log_dt_std

    def encode_delta(self, delta: np.ndarray) -> np.ndarray:
        return (delta - self.delta_mean) / self.delta_std

    def decode_delta(self, delta: np.ndarray) -> np.ndarray:
        return delta * self.delta_std + self.delta_mean


def generate_pairs(
    n_trajectories: int,
    n_steps: int,
    strides: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate supervised transitions with Cantera's ReactorNet solver."""
    rng = np.random.default_rng(seed)
    x_current: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    delta_y: list[np.ndarray] = []
    base_dt = T_TOTAL / n_steps

    for _ in range(n_trajectories):
        temperature = rng.uniform(1400.0, 1600.0)
        phi = rng.uniform(0.85, 1.15)
        state = solve_lagrangian(temperature, phi, n_steps)["state"]

        for stride in strides:
            x_current.append(state[:-stride])
            dt_values.append(
                np.full((len(state) - stride, 1), stride * base_dt)
            )
            delta_y.append(state[stride:, 1:] - state[:-stride, 1:])

    return (
        np.concatenate(x_current).astype(np.float32),
        np.concatenate(dt_values).astype(np.float32),
        np.concatenate(delta_y).astype(np.float32),
    )


def train(
    *,
    epochs: int = 250,
    n_train_trajectories: int = 20,
    n_valid_trajectories: int = 4,
    n_steps: int = 300,
    strides: tuple[int, ...] = (1, 2, 4, 8),
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    seed: int = 7,
    output_dir: str | Path = ".",
) -> tuple[DeepONetStepper, Normalizer, dict[str, list[float]]]:
    """Generate Cantera data, train the DeepONet, plot loss, and return history."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, dt_train, dy_train = generate_pairs(
        n_train_trajectories, n_steps, strides, seed
    )
    x_valid, dt_valid, dy_valid = generate_pairs(
        n_valid_trajectories, n_steps, strides, seed + 1
    )

    log_dt = np.log10(dt_train)
    normalizer = Normalizer(
        state_mean=x_train.mean(axis=0),
        state_std=np.maximum(x_train.std(axis=0), 1e-8),
        delta_mean=dy_train.mean(axis=0),
        delta_std=np.maximum(dy_train.std(axis=0), 1e-10),
        log_dt_mean=float(log_dt.mean()),
        log_dt_std=max(float(log_dt.std()), 1e-8),
    )

    def tensors(x: np.ndarray, dt: np.ndarray, dy: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(normalizer.encode_state(x).astype(np.float32)),
            torch.from_numpy(normalizer.encode_dt(dt).astype(np.float32)),
            torch.from_numpy(normalizer.encode_delta(dy).astype(np.float32)),
        )

    train_loader = DataLoader(
        tensors(x_train, dt_train, dy_train),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        tensors(x_valid, dt_valid, dy_valid),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONetStepper(x_train.shape[1], dy_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x, dt, target in train_loader:
            x, dt, target = x.to(device), dt.to(device), target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, dt), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        history["loss"].append(total / count)

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, dt, target in valid_loader:
                x, dt, target = x.to(device), dt.to(device), target.to(device)
                loss = loss_fn(model(x, dt), target)
                total += loss.item() * len(x)
                count += len(x)
        history["val_loss"].append(total / count)

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(
                f"epoch {epoch + 1:4d} | "
                f"loss {history['loss'][-1]:.3e} | "
                f"val {history['val_loss'][-1]:.3e}"
            )

    plt.figure(figsize=(6, 4))
    plt.semilogy(history["loss"], label="Training")
    plt.semilogy(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pfr_deeponet_loss.png", dpi=200)
    plt.close()

    return model, normalizer, history


@torch.no_grad()
def rollout_deeponet(
    model: DeepONetStepper,
    normalizer: Normalizer,
    temperature: float,
    phi: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Advance the PFR externally using only DeepONet forward passes."""
    gas = ct.Solution(REACTION_MECHANISM)
    set_inlet(gas, temperature, phi)
    initial_enthalpy = gas.enthalpy_mass
    mass_flow_rate = U0 * gas.density * AREA
    h2_index = gas.species_index("H2")

    dt = T_TOTAL / n_steps
    times = np.arange(n_steps + 1, dtype=float) * dt
    z = np.zeros(n_steps + 1)
    states = np.empty((n_steps + 1, gas.n_species + 1))
    x_h2 = np.empty(n_steps + 1)
    states[0] = np.r_[gas.T, gas.Y]
    x_h2[0] = gas.X[h2_index]

    device = next(model.parameters()).device
    model.eval()

    for i in range(1, n_steps + 1):
        encoded_state = torch.tensor(
            normalizer.encode_state(states[i - 1]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        encoded_dt = torch.tensor(
            normalizer.encode_dt(np.array([[dt]], dtype=np.float32)),
            dtype=torch.float32,
            device=device,
        )

        encoded_delta = model(encoded_state, encoded_dt).cpu().numpy()[0]
        delta_y = normalizer.decode_delta(encoded_delta)
        y_next = np.clip(states[i - 1, 1:] + delta_y, 0.0, None)
        y_next /= y_next.sum()

        # Adiabatic, constant-pressure correction: infer T from h, P, and Y.
        gas.HPY = initial_enthalpy, PRESSURE, y_next
        states[i] = np.r_[gas.T, gas.Y]
        x_h2[i] = gas.X[h2_index]
        velocity = mass_flow_rate / (AREA * gas.density)
        z[i] = z[i - 1] + velocity * dt

    return {"t": times, "z": z, "state": states, "X_H2": x_h2}


def plot_test_comparison(
    model: DeepONetStepper,
    normalizer: Normalizer,
    *,
    temperature: float = 1500.0,
    phi: float = 1.0,
    n_steps: int = 300,
    output_dir: str | Path = ".",
) -> None:
    """Compare the two traditional PFR methods with the DeepONet rollout."""
    output_dir = Path(output_dir)
    reference = solve_lagrangian(temperature, phi, n_steps)
    chain = solve_reactor_chain(temperature, phi, n_steps)
    learned = rollout_deeponet(model, normalizer, temperature, phi, n_steps)

    gas = ct.Solution(REACTION_MECHANISM)
    h2 = gas.species_index("H2")

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    plt.plot(reference["z"], reference["state"][:, 0], label="Cantera Lagrangian")
    plt.plot(chain["z"], chain["state"][:, 0], "--", label="Cantera reactor chain")
    plt.plot(learned["z"], learned["state"][:, 0], ":", label="DeepONet")
    plt.xlabel("z [m]")
    plt.ylabel("T [K]")
    plt.legend()

    plt.subplot(1, 2, 2)
    reference_x_h2 = np.array([
        _mass_to_h2_mole_fraction(gas, row[0], row[1:], h2)
        for row in reference["state"]
    ])
    chain_x_h2 = np.array([
        _mass_to_h2_mole_fraction(gas, row[0], row[1:], h2)
        for row in chain["state"]
    ])
    plt.plot(reference["t"], reference_x_h2, label="Cantera Lagrangian")
    plt.plot(chain["t"], chain_x_h2, "--", label="Cantera reactor chain")
    plt.plot(learned["t"], learned["X_H2"], ":", label="DeepONet")
    plt.xlabel("t [s]")
    plt.ylabel("X_H2 [-]")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pfr_deeponet_test.png", dpi=200)
    plt.close()


def _mass_to_h2_mole_fraction(
    gas: ct.Solution,
    temperature: float,
    mass_fractions: np.ndarray,
    h2_index: int,
) -> float:
    gas.TPY = temperature, PRESSURE, mass_fractions
    return float(gas.X[h2_index])


if __name__ == "__main__":
    out = Path("pfr_deeponet_results")
    model, normalizer, history = train(output_dir=out)
    plot_test_comparison(model, normalizer, output_dir=out)
    print(f"Saved plots to {out.resolve()}")
