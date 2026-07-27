from sympy import Function, Number, Symbol
import torch

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from reactor_physicsnemo_common import rel_l2, save_comparison, train_fno


class PackedBed(PDE):
    def __init__(self, da, alpha, eps):
        self.dim = 1
        w = Symbol("x"); X = Function("X")(w); y = Function("y")(w)
        self.equations = {
            "conversion": X.diff(w) - Number(da) * (1 - X) * y / (1 + Number(eps) * X),
            "pressure": y.diff(w) + Number(alpha) * (1 + Number(eps) * X) / (2 * y),
        }


def solve(w, da, alpha, eps):
    X = torch.zeros(w.shape[0], 1, device=w.device); y = torch.ones_like(X)
    Xs, ys = [X.clone()], [y.clone()]
    for i in range(1, w.shape[1]):
        h = w[:, i:i+1] - w[:, i-1:i]
        rX = da * (1 - X).clamp_min(0) * y / (1 + eps * X)
        ry = -alpha * (1 + eps * X) / (2 * y.clamp_min(.08))
        X = (X + h * rX).clamp(0, .995); y = (y + h * ry).clamp_min(.08)
        Xs.append(X.clone()); ys.append(y.clone())
    return torch.cat(Xs, 1).unsqueeze(-1), torch.cat(ys, 1).unsqueeze(-1)


def operator_data(n, m, device):
    w = torch.linspace(0, 1, m, device=device); ww = w[None, :, None].repeat(n, 1, 1)
    p = torch.cat([.4 + 1.8 * torch.rand(n, 1, device=device), .04 + .28 * torch.rand(n, 1, device=device), -.15 + .45 * torch.rand(n, 1, device=device)], 1)
    X, y = solve(ww[:, :, 0], p[:, 0:1], p[:, 1:2], p[:, 2:3])
    return w, p, torch.cat([ww, p[:, None, :].repeat(1, m, 1)], -1), torch.cat([X, y], -1)


def state(raw):
    return torch.cat([torch.sigmoid(raw[:, 0:1]), .08 + torch.nn.functional.softplus(raw[:, 1:2])], 1)


def main():
    torch.manual_seed(5); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    da, alpha, eps = 1.2, .18, .15
    pinn = FullyConnected(in_features=1, out_features=2, num_layers=4, layer_size=64).to(device)
    physics = PhysicsInformer(["conversion", "pressure"], PackedBed(da, alpha, eps), "autodiff", device=device)
    opt = torch.optim.Adam(pinn.parameters(), 1e-3)
    for step in range(3000):
        w = torch.rand(256, 1, device=device, requires_grad=True); pred = state(pinn(w)); X, y = pred[:, 0:1], pred[:, 1:2]
        r = physics.forward({"coordinates": w, "x": w, "X": X, "y": y})
        init = state(pinn(torch.zeros(1, 1, device=device)))
        loss = (r["conversion"] ** 2).mean() + (r["pressure"] ** 2).mean() + 25 * (init - torch.tensor([[0., 1.]], device=device)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    grid, params, xop, yop = operator_data(144, 96, device)
    fno = train_fno(xop, yop)
    w = torch.linspace(0, 1, 250, device=device)[:, None]
    X, y = solve(w.T, torch.tensor([[da]], device=device), torch.tensor([[alpha]], device=device), torch.tensor([[eps]], device=device)); exact = torch.cat([X, y], -1)[0]
    case = torch.tensor([[da, alpha, eps]], device=device)
    with torch.no_grad():
        yp = state(pinn(w))
        yf = fno(torch.cat([w[None], case[:, None, :].repeat(1, len(w), 1)], -1).permute(0, 2, 1)).permute(0, 2, 1)[0]
    print(f"Packed bed relative L2: PINN={rel_l2(yp,exact):.2e}, FNO={rel_l2(yf,exact):.2e}")
    save_comparison("Packed Bed", w[:, 0], exact, yp, yf, None, "W/Wmax", "state", ["conversion X", "pressure ratio y"])


if __name__ == "__main__":
    main()


