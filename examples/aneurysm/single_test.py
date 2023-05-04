from math import pi
from math import sqrt

import numpy as np
import torch


def det_test(x, y, nu, dP, mu, sigma, scale, epochs, path1, device, caseIdx):

    R = (
        scale
        * 1
        / sqrt(2 * np.pi * sigma**2)
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )
    rInlet = 0.05
    yUp = rInlet - R
    xt = torch.FloatTensor(x).to(device)
    yt = torch.FloatTensor(y).to(device)
    xt = xt.view(len(xt), -1)
    yt = yt.view(len(yt), -1)
    scalet = scale * torch.ones_like(xt)
    Rt = torch.FloatTensor(yUp).to(device)
    Rt = Rt.view(len(Rt), -1)
    xt.requires_grad = True
    yt.requires_grad = True
    scalet.requires_grad = True
    net_in = torch.cat((xt, yt, scalet), 1)
    u_t = net2(net_in)
    v_t = net3(net_in)
    P_t = net4(net_in)
    u_hard = u_t * (Rt**2 - yt**2)
    v_hard = (Rt**2 - yt**2) * v_t
    L = 1
    xStart = 0
    xEnd = L
    P_hard = (
        (xStart - xt) * 0
        + dP * (xEnd - xt) / L
        + 0 * yt
        + (xStart - xt) * (xEnd - xt) * P_t
    )
    u_hard = u_hard.cpu().data.numpy()
    v_hard = v_hard.cpu().data.numpy()
    P_hard = P_hard.cpu().data.numpy()

    np.savez(
        path1 + str(int(caseIdx)) + "ML_WallStress_uvp",
        x_center=x,
        y_center=y,
        u_center=u_hard,
        v_center=v_hard,
        p_center=P_hard,
    )

    return u_hard, v_hard, P_hard
