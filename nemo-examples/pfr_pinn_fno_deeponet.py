from sympy import Function, Number, Symbol
import torch

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from reactor_physicsnemo_common import rel_l2, save_comparison, train_fno


class PFR(PDE):
    def __init__(self, da):
        self.dim = 1
        z = Symbol("x"); C = Function("C")(z)
        self.equations = {"pfr": C.diff(z) + Number(da) * C}


def exact(z, cin, da):
    return cin * torch.exp(-da * z)


def operator_data(n, m, device):
    z = torch.linspace(0, 1, m, device=device); zz = z[None, :, None].repeat(n, 1, 1)
    p = torch.cat([0.7 + .8 * torch.rand(n, 1, device=device), .25 + 2.25 * torch.rand(n, 1, device=device)], 1)
    y = exact(zz, p[:, None, 0:1], p[:, None, 1:2])
    return z, p, torch.cat([zz, p[:, None, :].repeat(1, m, 1)], -1), y


def main():
    torch.manual_seed(4); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cin, da = 1.0, 1.35
    pinn = FullyConnected(in_features=1, out_features=1, num_layers=4, layer_size=64).to(device)
    physics = PhysicsInformer(["pfr"], PFR(da), "autodiff", device=device)
    opt = torch.optim.Adam(pinn.parameters(), 1e-3)
    for step in range(2500):
        z = torch.rand(256, 1, device=device, requires_grad=True); C = pinn(z)
        r = physics.forward({"coordinates": z, "x": z, "C": C})["pfr"]
        loss = (r ** 2).mean() + 20 * (pinn(torch.zeros(1, 1, device=device)) - cin).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    grid, params, xop, yop = operator_data(128, 96, device)
    fno = train_fno(xop, yop)
    z = torch.linspace(0, 1, 250, device=device)[:, None]; y = exact(z, cin, da)
    case = torch.tensor([[cin, da]], device=device)
    with torch.no_grad():
        yp = pinn(z)
        yf = fno(torch.cat([z[None], case[:, None, :].repeat(1, len(z), 1)], -1).permute(0, 2, 1)).permute(0, 2, 1)[0]
    print(f"PFR relative L2: PINN={rel_l2(yp,y):.2e}, FNO={rel_l2(yf,y):.2e}")
    save_comparison("PFR", z[:, 0], y, yp, yf, None, "z/L", "concentration C")


if __name__ == "__main__":
    main()


