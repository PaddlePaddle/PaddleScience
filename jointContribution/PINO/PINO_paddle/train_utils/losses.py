import numpy as np
import paddle
import paddle.nn.functional as F

def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = paddle.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = paddle.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du

def darcy_loss(u, a):
    batchsize = u.shape[0]
    size = u.shape[1]
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    # index_x = paddle.cat([paddle.tensor(range(0, size)), (size - 1) * paddle.ones(size), paddle.tensor(range(size-1, 1, -1)),
    #                      paddle.zeros(size)], dim=0).long()
    # index_y = paddle.cat([(size - 1) * paddle.ones(size), paddle.tensor(range(size-1, 1, -1)), paddle.zeros(size),
    #                      paddle.tensor(range(0, size))], dim=0).long()

    # boundary_u = u[:, index_x, index_y]
    # truth_u = paddle.zeros(boundary_u.shape, device=u.device)
    # loss_u = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = paddle.ones(Du.shape)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = paddle.mean(loss_f)
    return loss_f

def FDM_NS_vorticity(w, v=1/40, t_interval=1.0):
    batchsize = w.shape[0]
    nx = w.shape[1]
    ny = w.shape[2]
    nt = w.shape[3]
    w = w.reshape([batchsize, nx, ny, nt])

    w_h = paddle.fft.fft2(w, axes=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = paddle.concat((paddle.arange(start=0, end=k_max, step=1),
                     paddle.arange(start=-k_max, end=0, step=1)), 0).reshape([N, 1]).tile([1, N]).reshape([1,N,N,1])
    k_y = paddle.concat((paddle.arange(start=0, end=k_max, step=1),
                     paddle.arange(start=-k_max, end=0, step=1)), 0).reshape([1, N]).tile([N, 1]).reshape([1,N,N,1])
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = paddle.fft.irfft2(ux_h[:, :, :k_max + 1], axes=[1, 2])
    uy = paddle.fft.irfft2(uy_h[:, :, :k_max + 1], axes=[1, 2])
    wx = paddle.fft.irfft2(wx_h[:, :, :k_max+1], axes=[1,2])
    wy = paddle.fft.irfft2(wy_h[:, :, :k_max+1], axes=[1,2])
    wlap = paddle.fft.irfft2(wlap_h[:, :, :k_max+1], axes=[1,2])

    dt = t_interval / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
    return Du1

def Autograd_Burgers(u, grid, v=1/100):
    from paddle.autograd import grad
    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut

def AD_loss(u, u0, grid, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = paddle.zeros(nx,).long()
        index_x = paddle.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = paddle.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = paddle.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)

        all_norms = (h**(self.d/self.p))*paddle.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = paddle.norm(x.reshape([num_examples,-1]) - y.reshape([num_examples,-1]), self.p, 1)
        y_norms = paddle.norm(y.reshape([num_examples,-1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms/y_norms)
            else:
                return paddle.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def FDM_Burgers(u, v, D=1):
    batchsize = u.shape[0]
    nt = u.shape[1]
    nx = u.shape[2]

    u = u.reshape([batchsize, nt, nx])
    dt = D / (nt-1)
    dx = D / (nx)

    u_h = paddle.fft.fft(u, axis=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = paddle.concat((paddle.arange(start=0, end=k_max, step=1, dtype='float32'),
                     paddle.arange(start=-k_max, end=0, step=1, dtype='float32')), 0).reshape([1,1,nx])
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    ux = paddle.fft.irfft(ux_h[:, :, :k_max+1], axis=2, n=nx)
    uxx = paddle.fft.irfft(uxx_h[:, :, :k_max+1], axis=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du

def PINO_loss(u, u0, v):
    batchsize = u.shape[0]
    nt = u.shape[1]
    nx = u.shape[2]

    u = u.reshape([batchsize, nt, nx])
    # lploss = LpLoss(size_average=True)

    index_t = paddle.zeros(1,'int32')
    index_x = paddle.to_tensor(list(range(nx)),'int32')
    boundary_u = paddle.index_select(u, index_t, axis=1).squeeze(1)
    # boundary_u = paddle.index_select(boundary_u, index_x, axis=2)
    # boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u, v)[:, :, :]
    f = paddle.zeros(Du.shape)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_f

def PINO_loss3d(u, u0, forcing, v=1/40, t_interval=1.0):
    batchsize = u.shape[0]
    nx = u.shape[1]
    ny = u.shape[2]
    nt = u.shape[3]

    u = u.reshape([batchsize, nx, ny, nt])
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.tile([batchsize, 1, 1, nt-2])
    loss_f = lploss(Du, f)

    return loss_ic, loss_f

def PDELoss(model, x, t, nu):
    '''
    Compute the residual of PDE:
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params:
        - model
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return:
        - mean of residual : scalar
    '''
    u = model(paddle.cat([x, t], dim=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = paddle.autograd.grad(outputs=[u.sum()], inputs=[x, t], create_graph=True)
    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = paddle.autograd.grad(outputs=[grad_x.sum()], inputs=[x], create_graph=True)

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual

def get_forcing(S):
    x1 = paddle.to_tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=paddle.float32).reshape([S, 1]).tile([1, S])
    x2 = paddle.to_tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=paddle.float32).reshape([1, S]).tile([S, 1])
    return -4 * (paddle.cos(4*(x2))).reshape([1,S,S,1])