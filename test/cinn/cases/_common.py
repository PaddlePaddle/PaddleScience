"""Helpers shared by the case definitions.

A case module describes one PaddleScience example with synthetic data. It must
define:

    BATCH_SIZE: int                       samples per step, used for throughput
    build_model() -> paddle.nn.Layer      same arch/hyper-params as examples/<case>/conf
    build_batch() -> (inputs, labels)     flat dicts of paddle.Tensor; inputs that
                                          need derivatives set stop_gradient=False
    compute_loss(model, inputs, labels)   scalar training loss (PDE residual included)

and may define:

    forward(model, inputs)                inference call, defaults to model(inputs)
    prepare(model)                        one-off setup before to_static, e.g. lambdify
    LEARNING_RATE: float                  Adam lr, defaults to 1e-3
    EXAMPLE: str                          examples/ sub-directory the case mirrors
    NOTE: str                             what was simplified compared with the example
"""
import paddle
import paddle.nn.functional as F

from ppsci import autodiff


def uniform(shape, low=0.0, high=1.0, requires_grad=False):
    t = paddle.uniform(shape, dtype="float32", min=low, max=high)
    t.stop_gradient = not requires_grad
    return t


def randn(shape, requires_grad=False):
    t = paddle.randn(shape, dtype="float32")
    t.stop_gradient = not requires_grad
    return t


def randint(shape, high, low=0):
    return paddle.randint(low, high, shape, dtype="int64")


def points(keys, n, low=0.0, high=1.0, prefix=""):
    """Collocation points `{prefix+key: [n, 1]}` that require gradients."""
    return {prefix + k: uniform([n, 1], low, high, requires_grad=True) for k in keys}


def take(inputs, keys, prefix=""):
    """Select `prefix+key` entries and strip the prefix, e.g. boundary batches."""
    return {k: inputs[prefix + k] for k in keys}


def mse(a, b):
    return F.mse_loss(a, b)


def sq_mean(x):
    return paddle.mean(paddle.square(x))


def data_loss(out, labels, keys=None):
    keys = keys or list(labels.keys())
    return sum(mse(out[k], labels[k]) for k in keys)


def lambdify(equations, model, names=None):
    """sympy equations of a ppsci.equation PDE -> {name: callable(data_dict)},
    converted the same way ppsci.solver.Solver does (fuse_derivative=True)."""
    import ppsci

    names = list(names or equations.keys())
    funcs = ppsci.lambdify([equations[n] for n in names], model, fuse_derivative=True)
    if not isinstance(funcs, list):
        funcs = [funcs]
    return dict(zip(names, funcs))


def pde_loss(model, inputs, equations, weights=None):
    """Mean squared residual of `equations` (name -> callable from lambdify).

    Mirrors ppsci.utils.expression.ExpressionSolver.train_forward: the model
    output and the inputs form one data dict that every expression reads, and
    the autodiff cache is cleared afterwards.
    """
    out = model(inputs)
    data = {**inputs, **out}
    loss = 0.0
    for name, expr in equations.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        loss = loss + w * sq_mean(expr(data))
    autodiff.clear()
    return loss


def bc_loss(model, inputs, targets):
    """Dirichlet loss: `targets` maps output key -> tensor or float."""
    out = model(inputs)
    loss = 0.0
    for k, v in targets.items():
        if not isinstance(v, paddle.Tensor):
            v = paddle.full_like(out[k], v)
        loss = loss + mse(out[k], v)
    return loss


def grad(y, x):
    return paddle.grad(y, x, create_graph=True)[0]
