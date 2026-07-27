from sympy import Function, Number, Symbol
import torch

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from reactor_physicsnemo_common import rel_l2, save_comparison, train_fno


class CSTR(PDE):
    def __init__(self, cin, tau, k):
        self.dim = 1
        t = Symbol("x"); C = Function("C")(t)
        self.equations = {"cstr": C.diff(t) - ((Number(cin) - C) / Number(tau) - Number(k) * C)}


def exact(t, cin, c0, tau, k):
    css = cin / (1 + k * tau)
    return css + (c0 - css) * torch.exp(-(1 / tau + k) * t)


def operator_data(n, m, device):
    t = torch.linspace(0, 5, m, device=device); tt = t[None, :, None].repeat(n, 1, 1)
    p = torch.cat([0.8 + .7 * torch.rand(n, 1, device=device), .05 + .8 * torch.rand(n, 1, device=device), .6 + 1.6 * torch.rand(n, 1, device=device), .15 + 1.15 * torch.rand(n, 1, device=device)], 1)
    y = exact(tt, p[:, None, 0:1], p[:, None, 1:2], p[:, None, 2:3], p[:, None, 3:4])
    return t, p, torch.cat([tt / 5, p[:, None, :].repeat(1, m, 1)], -1), y


def main():
    torch.manual_seed(3); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cin, c0, tau, k = 1.0, 0.15, 1.2, 0.65
    pinn = FullyConnected(in_features=1, out_features=1, num_layers=4, layer_size=64).to(device)
    physics = PhysicsInformer(["cstr"], CSTR(cin, tau, k), "autodiff", device=device)
    opt = torch.optim.Adam(pinn.parameters(), 1e-3)
    for step in range(2500):
        t = 5 * torch.rand(256, 1, device=device, requires_grad=True); C = pinn(t / 5)
        r = physics.forward({"coordinates": t, "x": t, "C": C})["cstr"]
        loss = (r ** 2).mean() + 20 * (pinn(torch.zeros(1, 1, device=device)) - c0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    grid, params, xop, yop = operator_data(128, 96, device)
    fno = train_fno(xop, yop)
    t = torch.linspace(0, 5, 250, device=device)[:, None]; y = exact(t, cin, c0, tau, k)
    case = torch.tensor([[cin, c0, tau, k]], device=device)
    with torch.no_grad():
        yp = pinn(t / 5)
        yf = fno(torch.cat([t[None] / 5, case[:, None, :].repeat(1, len(t), 1)], -1).permute(0, 2, 1)).permute(0, 2, 1)[0]
    print(f"CSTR relative L2: PINN={rel_l2(yp,y):.2e}, FNO={rel_l2(yf,y):.2e}")
    save_comparison("CSTR", t[:, 0], y, yp, yf, None, "time", "concentration C")


if __name__ == "__main__":
    main()


