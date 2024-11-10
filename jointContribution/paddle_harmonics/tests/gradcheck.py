# mypy: allow-untyped-defs
# code from pytorch
import collections
import functools
import warnings
from itertools import product
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from paddle_harmonics.utils import paddle_aux  # noqa


def is_tensor_like(obj):
    return isinstance(obj, paddle.Tensor)


class GradcheckError(RuntimeError):
    r"""Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""


def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def _allocate_jacobians_with_inputs(
    input_tensors: Tuple, numel_output
) -> Tuple[paddle.Tensor, ...]:
    # Makes zero-filled tensors from inputs. If `numel_output` is not None, for
    # each tensor in `input_tensors`, returns a new zero-filled tensor with height
    # of `t.numel` and width of `numel_output`. Otherwise, for each tensor, returns
    # a 1-d tensor with size `(t.numel,)`. Each new tensor will be strided and have
    # the same dtype and device as those of the corresponding input.
    out: List[paddle.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and not t.stop_gradient:
            out.append(paddle.zeros((t.size, numel_output)))
    return tuple(out)


def _allocate_jacobians_with_outputs(
    output_tensors: Tuple, numel_input, dtype=None, device=None
) -> Tuple[paddle.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor
    # in `output_tensors`, returns a new zero-filled tensor with height of `dim` and
    # width of `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size
    # (t.numel,).
    out: List[paddle.Tensor] = []
    options = {"dtype": dtype}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(paddle.zeros((numel_input, t.size), **options))
    return tuple(out)


def _iter_tensors(
    x: Union[paddle.Tensor, Iterable[paddle.Tensor]], only_requiring_grad: bool = False
) -> Iterable[paddle.Tensor]:
    if is_tensor_like(x):
        # mypy doesn't narrow type of `x` to paddle.Tensor
        if not x.stop_gradient or not only_requiring_grad:  # type: ignore[union-attr]
            yield x  # type: ignore[misc]
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)


def _densify(x):
    # return a copy of sparse x with all unspecified elements
    # "replaced" with zero-valued elements
    if isinstance(x, (list, tuple)):
        return type(x)(map(_densify, x))
    return x


def _iter_tensor(x_tensor):
    # Use .data here to get around the version check
    for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.shape])):
        yield x_tensor, x_idx, d_idx


def _get_numerical_jacobian(
    fn, inputs, outputs=None, target=None, eps=1e-3, is_forward_ad=False
) -> List[Tuple[paddle.Tensor, ...]]:
    """Compute the numerical Jacobian of `fn(inputs)` with respect to `target`.

    If not specified, targets are the input. Returns M * N Jacobians where N is the
    number of tensors in target that require grad and M is the number of non-integral
    outputs.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        is_forward_ad: if this numerical jacobian is computed to be checked wrt
                       forward AD gradients (this is used for error checking only)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: List[Tuple[paddle.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError(
            "Expected output to be non-complex. get_numerical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    if target is None:
        target = inputs
    inp_indices = [i for i, a in enumerate(target) if is_tensor_like(a) and not a.stop_gradient]

    for i, (inp, inp_idx) in enumerate(zip(_iter_tensors(target, True), inp_indices)):
        jacobians += [
            get_numerical_jacobian_wrt_specific_input(
                fn,
                inp_idx,
                inputs,
                outputs,
                eps,
                input=inp,
                is_forward_ad=is_forward_ad,
            )
        ]
    return jacobians


def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    This is a Deprecated API.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        inputs: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        grad_out: defaults to 1.0.

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_numerical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )

    def fn_pack_inps(*inps):
        return fn(inps)

    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    # Computes numerical directional derivative as finite difference
    # of function `fn` at input `entry`, perturbed by vector `v`.

    orig = entry.clone()
    paddle.assign(orig - v, entry)
    outa = fn()
    paddle.assign(orig + v, entry)
    outb = fn()
    paddle.assign(orig, entry)

    def compute(a, b):
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)  # use central difference approx
        return ret.detach().reshape([-1])

    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


def _compute_numerical_jvps_wrt_specific_input(
    jvp_fn, delta, input_is_complex, is_forward_ad=False
) -> List[paddle.Tensor]:
    # Computing the jacobian only works for real delta
    # For details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jvps: List[paddle.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)

    if input_is_complex:  # C -> R
        ds_dy_tup = jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            assert not ds_dx.is_complex()
            # conjugate wirtinger derivative
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # R -> R or (R -> C for the forward AD case)
            assert is_forward_ad or not ds_dx.is_complex()
            jvps.append(ds_dx)
    return jvps


def _combine_jacobian_cols(
    jacobians_cols: Dict[int, List[paddle.Tensor]], outputs, input, numel
) -> Tuple[paddle.Tensor, ...]:
    # jacobian_cols maps column_idx -> output_idx -> single column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = _allocate_jacobians_with_outputs(
        outputs, numel, dtype=input.dtype if paddle.is_complex(input) else None
    )
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


def _prepare_input(
    input: paddle.Tensor, maybe_perturbed_input: Optional[paddle.Tensor], fast_mode=False
) -> paddle.Tensor:
    return input


def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    # Check that the returned outputs don't have different dtype or shape when you
    # perturb the input
    on_index = "on index {idx} " if idx is not None else ""
    assert output1.shape == output2.shape, (
        f"Expected `func` to return outputs with the same shape"
        f" when inputs are perturbed {on_index}by {eps}, but got:"
        f" shapes {output1.shape} and {output2.shape}."
    )
    assert output1.dtype == output2.dtype, (
        f"Expected `func` to return outputs with the same dtype"
        f" when inputs are perturbed {on_index}by {eps}, but got:"
        f" dtypes {output1.dtype} and {output2.dtype}."
    )


def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False
) -> Tuple[paddle.Tensor, ...]:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs. We use a dictionary for
    # jacobian_cols because indices aren't necessarily consecutive for sparse inputs
    # When we perturb only a single element of the input tensor at a time, the jvp
    # is equivalent to a single col of the Jacobian matrix of fn.
    jacobian_cols: Dict[int, List[paddle.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    assert not input.stop_gradient
    for x, idx, d_idx in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(
            jvp_fn, eps, x.is_complex(), is_forward_ad
        )
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def _get_analytical_jacobian_forward_ad(
    fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None
) -> Tuple[Tuple[paddle.Tensor, ...], ...]:
    """Compute the analytical Jacobian using forward mode AD of `fn(inputs)` using forward mode AD with respect to `target`.

    Return N * M Jacobians where N is the number of tensors in target that require grad and
    M is the number of non-integral outputs.
    Contrary to other functions here, this function requires "inputs" to actually be used by the function.
    The computed value is expected to be wrong if the function captures the inputs by side effect instead of
    using the passed ones (many paddle.nn tests do this).

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    # To avoid early import issues
    fwAD = autograd.forward_ad  # noqa

    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and not i.stop_gradient)

    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError(
            "Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad."
        )

    if all_u:
        jacobians = tuple(_allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs)
    else:
        jacobians = tuple(
            _allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs
        )

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and not inp.stop_gradient:
                inp = fwAD.make_dual(inp.detach(), paddle.zeros(inp.shape, dtype=inp.dtype))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)

        if all_u:
            # Do the full reduction in one pass
            # To be consistent with numerical evaluation, we actually compute one reduction per input
            for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
                # fw_grad.copy_(u.view_as(fw_grad))
                paddle.assign(u.view_as(fw_grad), fw_grad)
                raw_outputs = _as_tuple(fn(*dual_inputs))
                dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                for index_o, d_o in enumerate(dual_outputs):
                    val, res = fwAD.unpack_dual(d_o)
                    if (
                        check_grad_dtypes
                        and res is not None
                        and val.is_complex() != res.is_complex()
                    ):
                        raise GradcheckError("Forward AD gradient has dtype mismatch.")

                    # Remove extra dimension of size 1 corresponding to the reduced input
                    jacobians[i][index_o].squeeze_(0)
                    if res is None:
                        jacobians[i][index_o].zero_()
                    else:
                        # jacobians[i][index_o].copy_(res.reshape([-1]))
                        paddle.assign(res.reshape([-1]), jacobians[i][index_o])
                fw_grad.zero_()
        else:
            # Reconstruct the full Jacobian column by column
            for i, fw_grad in enumerate(fw_grads):
                for lin_idx, grad_idx in enumerate(product(*[range(m) for m in fw_grad.size()])):
                    fw_grad[grad_idx] = 1.0
                    raw_outputs = _as_tuple(fn(*dual_inputs))
                    dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                    for index_o, d_o in enumerate(dual_outputs):
                        val, res = fwAD.unpack_dual(d_o)
                        if (
                            check_grad_dtypes
                            and res is not None
                            and val.is_complex() != res.is_complex()
                        ):
                            raise GradcheckError("Forward AD gradient has dtype mismatch.")

                        if res is None:
                            jacobians[i][index_o][lin_idx].zero_()
                        else:
                            # jacobians[i][index_o][lin_idx].copy_(res.reshape([-1]))
                            paddle.assign(res.reshape([-1]), jacobians[i][index_o][lin_idx])
                    fw_grad[grad_idx] = 0.0

    return jacobians


def _get_input_to_perturb(input):
    input_to_perturb = input.data
    return input_to_perturb


def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    # Wraps `fn` so that its inputs are already supplied
    def wrapped_fn():
        inp = tuple(
            _prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode)
            if is_tensor_like(a)
            else a
            for i, a in enumerate(_as_tuple(inputs))
        )
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    return wrapped_fn


def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    # Wraps jvp_fn so that certain arguments are already supplied
    def jvp_fn(delta):
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)

    return jvp_fn


def _reshape_tensor_or_tuple(u, shape):
    # We don't need to reshape when input corresponding to u is sparse
    if isinstance(u, tuple):
        return (u[0].reshape(shape), u[1].reshape(shape))
    else:
        return u.reshape(shape)


def _mul_tensor_or_tuple(u, k):
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    else:
        return k * u


def _get_numerical_jvp_wrt_specific_input(
    fn, input_idx, inputs, u, eps, is_forward_ad=False
) -> List[paddle.Tensor]:
    input = inputs[input_idx]
    input_to_perturb = _get_input_to_perturb(input)
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    return _compute_numerical_jvps_wrt_specific_input(jvp_fn, u, input.is_complex(), is_forward_ad)


def _check_jacobians_equal(j1, j2, atol):
    # Check whether the max difference between two Jacobian tensors are within some
    # tolerance `atol`.
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def _stack_and_check_tensors(
    list_of_list_of_tensors, inputs, numel_outputs
) -> Tuple[Tuple[paddle.Tensor, ...], bool, bool]:
    # For the ith tensor in the inner list checks whether it has the same size and
    # dtype as the ith differentiable input.
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(_iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, tensor_list in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, tensor in enumerate(tensor_list):
            if tensor is not None and tuple(tensor.shape) != tuple(inp.shape):
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                out_jacobian[:, j].zero_()
            else:
                dense = tensor
                assert out_jacobian[:, j].numel() == dense.numel()
                out_jacobian[:, j] = dense.reshape([-1])
    return out_jacobians, correct_grad_sizes, correct_grad_types


FAILED_NONDET_MSG = """\n
If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `nondet_tol=<tol>` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_nondet_tol=<tol>`.
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_nondet_tol=<tol>`
"""


def _check_analytical_jacobian_attributes(
    inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None
) -> Tuple[paddle.Tensor, ...]:
    # This is used by both fast and slow mode:
    #  - For slow mode, vjps[i][j] is the jth row of the Jacobian wrt the ith
    #    input.
    #  - For fast mode, vjps[i][0] is a linear combination of the rows
    #    of the Jacobian wrt the ith input
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return paddle.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel() if not fast_mode else 1
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    if not types_ok and check_grad_dtypes:
        raise GradcheckError("Gradient has dtype mismatch")
    if not sizes_ok:
        raise GradcheckError("Analytical gradient has incorrect size")
    if not reentrant:
        raise GradcheckError(
            "Backward is not reentrant, i.e., running backward with "
            "same input and grad_output multiple times gives different values, "
            "although analytical gradient matches numerical gradient."
            f"The tolerance for nondeterminism was {nondet_tol}." + FAILED_NONDET_MSG
        )
    return jacobians1


def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    # This shares much of its code with _check_analytical_jacobian_attributes
    if grad_out != 1.0:  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_analytical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )
    if output.is_complex():
        raise ValueError(
            "Expected output to be non-complex. get_analytical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return paddle.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(vjps1, inputs, output_numel)
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    return jacobians1, reentrant, sizes_ok, types_ok


def _compute_analytical_jacobian_rows(vjp_fn, sample_output) -> List[List[Optional[paddle.Tensor]]]:
    # Computes Jacobian row-by-row by projecting `vjp_fn` = v^T J on standard basis
    # vectors: vjp_fn(e) = e^T J is a corresponding row of the Jacobian.
    # NB: this function does not assume vjp_fn(v) to return tensors with the same
    # number of elements for different v. This is checked when we later combine the
    # rows into a single tensor.
    grad_out_base = paddle.zeros_like(sample_output)
    # Note: After reshaping, the data may not be in the same memory location,
    # and it needs to be reshaped again to use the same memory location.
    flat_grad_out = grad_out_base.reshape([-1])
    grad_out_base = flat_grad_out.reshape(grad_out_base.shape)

    # jacobians_rows[i][j] is the Jacobian jth row for the ith input
    jacobians_rows: List[List[Optional[paddle.Tensor]]] = []
    for j in range(flat_grad_out.numel()):
        # flat_grad_out.zero_()
        flat_grad_out[j] = 1.0  # projection for jth row of Jacobian
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [d_x.clone() if isinstance(d_x, paddle.Tensor) else None]
    return jacobians_rows


def _check_inputs(tupled_inputs) -> bool:
    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and not inp.stop_gradient:
            if not (inp.dtype == paddle.float64 or inp.dtype == paddle.complex128):
                warnings.warn(
                    f"Input #{idx} requires gradient and "
                    "is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are "
                    "not of double precision floating point or complex. "
                )

            any_input_requiring_grad = True

    if not any_input_requiring_grad:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of the them have stop_gradient=False."
        )
    return True


def _check_outputs(outputs) -> None:
    return


def _check_no_differentiable_outputs(func, inputs, func_out, eps, *, is_forward_ad) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(
        func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad
    )
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if paddle.not_equal(jacobian, 0).sum() > 0:
                raise GradcheckError("Numerical gradient for function expected to be zero")
    return True


def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: List[paddle.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")
    grads_input = paddle.grad(
        outputs,
        diff_input_list,
        [paddle.zeros_like(o) for o in outputs],
        allow_unused=True,
    )
    for gi, di in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if masked:
            if not paddle.allclose(gi, paddle.zeros_like(gi)):
                raise GradcheckError("backward not multiplied by grad_output")
        elif not gi.equal(y=0).astype("bool").all():
            raise GradcheckError("backward not multiplied by grad_output")
        if gi.dtype != di.dtype:
            raise GradcheckError("grad is incorrect type")
        # if gi.device != di.device:
        #     raise GradcheckError("grad is incorrect device")
        if gi.shape != di.shape:
            raise GradcheckError("grad is incorrect size")
    return True


def _test_undefined_forward_mode(func, outputs, inputs):
    return True
    # fwAD = autograd.forward_ad

    # inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    # all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=True)

    # tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and not i.stop_gradient)

    # with fwAD.dual_level():
    #     fw_grads = []
    #     dual_inputs = []
    #     tensor_indices = set()
    #     for i, inp in enumerate(inputs):
    #         if is_tensor_like(inp) and not inp.stop_gradient:
    #             inp = fwAD.make_dual(inp.detach(), paddle.zeros_like(inp))
    #             # If inp is a differentiable view, the dual might not be the tangent given to
    #             # make_dual, so read it explicitly from the dual tensor
    #             fw_grads.append(fwAD.unpack_dual(inp)[1])
    #             tensor_indices.add(i)
    #         dual_inputs.append(inp)

    #     for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
    #         fw_grad.copy_(u.view_as(fw_grad))

    #     for idx, inp in enumerate(inputs):
    #         if idx not in tensor_indices:
    #             continue
    #         dual_inp_obj = dual_inputs[idx]

    #         # case 1 (Materialized Zero Tensor Tangent)
    #         dual_inputs[idx] = fwAD.make_dual(inp.detach(), paddle.zeros_like(inp))
    #         raw_outputs = _as_tuple(func(*dual_inputs))
    #         dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)

    #         # case 2 (Efficient Zero Tensor Tangent since we don't make a dual object and pass a regular tensor)
    #         dual_inputs[idx] = inp.detach()
    #         raw_outputs = _as_tuple(func(*dual_inputs))
    #         dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)

    #         # reset
    #         dual_inputs[idx] = dual_inp_obj

    #         for index_o, (d_o1, d_o2) in enumerate(zip(dual_outputs1, dual_outputs2)):
    #             val1, res1 = fwAD.unpack_dual(d_o1)
    #             val2, res2 = fwAD.unpack_dual(d_o2)

    #             if not (res1 is None or res2 is None):
    #                 if not paddle.allclose(res1, res2):
    #                     raise GradcheckError(
    #                         "Mismatch in tangent values for output with index: ",
    #                         index_o,
    #                         " when input: ",
    #                         inp,
    #                         " has an undefined tangent value. ",
    #                         " Got: ",
    #                         res1,
    #                         " but expected: ",
    #                         res2,
    #                     )
    # return True


def _test_undefined_backward_mode(func, outputs, inputs) -> bool:
    diff_input_list: List[paddle.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")

    def warn_bc_breaking():
        warnings.warn(
            "Backwards compatibility: New undefined gradient support checking "
            "feature is enabled by default, but it may break existing callers "
            "of this function. If this is true for you, you can call this "
            'function with "check_undefined_grad=False" to disable the feature'
        )

    def check_undefined_grad_support(output_to_check):
        grads_output = [paddle.zeros_like(o) for o in output_to_check]
        try:
            grads_input = paddle.grad(
                output_to_check, diff_input_list, grads_output, allow_unused=True
            )
        except RuntimeError as e:
            warn_bc_breaking()
            raise GradcheckError(
                "Expected backward function to handle undefined output grads. "
                'Please look at "Notes about undefined output gradients" in '
                '"tools/autograd/derivatives.yaml"'
            ) from e

        for gi, i in zip(grads_input, diff_input_list):
            if (gi is not None) and (not gi.equal(y=0).astype("bool").all()):
                warn_bc_breaking()
                raise GradcheckError(
                    "Expected all input grads to be undefined or zero when all output grads are undefined "
                    'or zero. Please look at "Notes about undefined output gradients" in '
                    '"tools/autograd/derivatives.yaml"'
                )
        return True

    # All backward functions must work properly if all output grads are undefined
    outputs_to_check = [
        [
            # paddle._C._functions.UndefinedGrad()(o)
            o
            for o in _differentiable_outputs(func(*inputs))
            # This check filters out Tensor-likes that aren't instances of Tensor.
            if isinstance(o, paddle.Tensor)
        ]
    ]

    # If there are multiple output grads, we should be able to undef one at a time without error
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append(
                [
                    # paddle._C._functions.UndefinedGrad()(o)
                    o if idx == undef_grad_idx else o
                    for idx, o in enumerate(output_to_check)
                ]
            )

    return all(check_undefined_grad_support(output) for output in outputs_to_check)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if not o.stop_gradient)


def _get_notallclose_msg(
    analytical,
    numerical,
    output_idx,
    input_idx,
    complex_indices,
    test_imag=False,
    is_forward_ad=False,
) -> str:
    out_is_complex = (not is_forward_ad) and complex_indices and output_idx in complex_indices
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    prefix = (
        ""
        if not (out_is_complex or inp_is_complex)
        else f"While considering the {part} part of complex {element} only, "
    )
    mode = "computed with forward mode " if is_forward_ad else ""
    return (
        prefix + "Jacobian %smismatch for output %d with respect to input %d,\n"
        "numerical:%s\nanalytical:%s\n" % (mode, output_idx, input_idx, numerical, analytical)
    )


def _transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def _real_and_imag_output(fn):
    # returns new functions real(fn), and imag(fn) where real(fn) and imag(fn) behave the same as
    # the original fn, except real or imag are applied to the complex outputs
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)

        return wrapped_fn

    return apply_to_c_outs(fn, paddle.real), apply_to_c_outs(fn, paddle.imag)


def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):
    # returns new functions that take real inputs instead of complex inputs as
    # (x, y) -> fn(x + y * 1j). And it computes: inp -> fn(inp + y * 1j) and inp -> fn(x + inp * 1j).
    # In each case, the other part is considered constant.
    # We do not use 0 for the constant here to make sure we always call the user function with a valid input.
    def apply_to_c_inps(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(
                    new_inputs[should_be_complex], tupled_inputs[should_be_complex]
                )
            return _as_tuple(fn(*new_inputs))

        return wrapped_fn

    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return real_fn, imag_fn


def _gradcheck_real_imag(
    gradcheck_fn,
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    check_backward_ad,
    nondet_tol,
    check_undefined_grad,
):
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any(o.is_complex() for o in _as_tuple(func_out))
    if check_backward_ad:
        if has_any_complex_output:
            real_fn, imag_fn = _real_and_imag_output(func)

            imag_func_out = imag_fn(*tupled_inputs)
            imag_outputs = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                tupled_inputs,
                imag_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
                test_imag=True,
            )

            real_func_out = real_fn(*tupled_inputs)
            real_outputs = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                tupled_inputs,
                real_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
            )
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
            )

        complex_inp_indices = [
            i for i, inp in enumerate(tupled_inputs) if is_tensor_like(inp) and inp.is_complex()
        ]
        if complex_inp_indices:
            real_fn, imag_fn = _real_and_imag_input(func, complex_inp_indices, tupled_inputs)

            imag_inputs = [
                inp.imag if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                imag_inputs,
                diff_imag_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                test_imag=True,
                use_forward_ad=True,
            )

            real_inputs = [
                inp.real if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                real_inputs,
                diff_real_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(func, outputs, tupled_inputs)


def _slow_gradcheck(
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    func_out = _as_tuple(func_out)
    if not outputs:
        return _check_no_differentiable_outputs(
            func, tupled_inputs, func_out, eps=eps, is_forward_ad=use_forward_ad
        )
    tupled_inputs_numerical = tupled_inputs if masked else _densify(tupled_inputs)

    numerical = _transpose(
        _get_numerical_jacobian(
            func,
            tupled_inputs_numerical,
            func_out,
            eps=eps,
            is_forward_ad=use_forward_ad,
        )
    )

    # Note: [numerical vs analytical output length]
    # The numerical path returns jacobian quantity for all outputs, even if not stop_gradient of that
    # output is False. This behavior is necessary for _check_no_differentiable_outputs to work.
    numerical = [nj for o, nj in zip(func_out, numerical) if not o.stop_gradient]
    if use_forward_ad:
        # analytical_forward = _get_analytical_jacobian_forward_ad(
        #     func, tupled_inputs, func_out, check_grad_dtypes=check_grad_dtypes
        # )

        # for i, n_per_out in enumerate(numerical):
        #     for j, n in enumerate(n_per_out):
        #         a = analytical_forward[j][i]
        #         if not _allclose_with_type_promotion(a, n.to(a.place), rtol, atol):
        #             raise GradcheckError(
        #                 _get_notallclose_msg(
        #                     a, n, i, j, complex_indices, test_imag, is_forward_ad=True
        #                 )
        #             )
        return True
    else:
        for i, o in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(
                tupled_inputs, o, nondet_tol, check_grad_dtypes
            )

            for j, (a, n) in enumerate(zip(analytical, numerical[i])):
                if not _allclose_with_type_promotion(a, n.to(a.place), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(a, n, i, j, complex_indices, test_imag)
                    )

    return True


def _allclose_with_type_promotion(a, b, rtol, atol):
    return paddle.allclose(a, b, rtol, atol)


def _to_real_dtype(dtype):
    if dtype == paddle.complex128:
        return paddle.float64
    elif dtype == paddle.complex64:
        return paddle.float32
    else:
        return dtype


def _vec_from_tensor(x, generator=None, downcast_complex=False):
    # Create a random vector with the same number of elements as x and the same
    # dtype/device. If x is complex and downcast_complex is False, we create a
    # complex tensor with only real component.

    dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
    vec = paddle.rand(x.numel()).to(dtype=dtype, device=x.device)
    vec /= vec.norm()
    return vec


def _get_inp_tensors(tupled_inputs):
    inp_idx_tup = [
        (i, t) for i, t in enumerate(tupled_inputs) if is_tensor_like(t) and not t.stop_gradient
    ]
    return [tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup]


def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    # Use our own generator to avoid messing with the user's RNG state
    g_cpu = None

    def _vec_from_tensor_cpu(*args):
        # Default allocate all tensors on CPU, so they are on the same device as the generator
        # even if the user specified a default device
        return _vec_from_tensor(*args)

    all_u = []
    all_u_dense = []
    for inp in inp_tensors:
        ur = _vec_from_tensor_cpu(inp, g_cpu, True)
        ur_dense = ur
        if inp.is_complex():
            ui = _vec_from_tensor_cpu(inp, g_cpu, True)
            all_u.append((ur, ui))
            ui_dense = ui
            all_u_dense.append((ur_dense, ui_dense))
        else:
            all_u.append(ur)
            all_u_dense.append(ur_dense)
    all_v = None if use_forward_ad else [_vec_from_tensor_cpu(out, g_cpu) for out in outputs]
    return all_v, all_u, all_u_dense


# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressable in the type system at the moment.
# If https://mypy.readthedocs.io/en/latest/additional_features.html?highlight=callable#extended-callable-types is accepted,
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.
def gradcheck(
    func,  # See Note [VarArg of Tensors]
    inputs,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_backward_ad: bool = True,
    masked: Optional[bool] = None,
) -> bool:  # noqa: D400,D205
    r"""Check gradients computed via small finite differences against analytical
    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
    and with ``stop_gradient=False``.

    The check between numerical and analytical gradients uses :func:`~paddle.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. note::
        Gradcheck may fail when evaluated on non-differentiable points
        because the numerically computed gradients via finite differencing may differ
        those computed analytically (not necessarily because either is incorrect).
        For more context, see :ref:`non-differentiable-func-grad`.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`paddle.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, optional): if ``True``, check if undefined output grads
            are supported and treated as zeros, for ``Tensor`` outputs.
        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on
            backward mode AD to be implemented. Defaults to ``True``.
        masked (bool, optional): if ``True``, the gradients of unspecified elements of
            sparse tensors are ignored. Defaults to ``False``.
    Returns:
        ``True`` if all differences satisfy allclose condition

    """
    args = locals().copy()
    args.pop("raise_exception")
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)
        except GradcheckError:
            return False
    else:
        return _gradcheck_helper(**args)


def _gradcheck_helper(
    func,
    inputs,
    eps,
    atol,
    rtol,
    nondet_tol,
    check_undefined_grad,
    check_grad_dtypes,
    check_backward_ad,
    masked,
):
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs)

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)

    gradcheck_fn = functools.partial(_slow_gradcheck, masked=masked)

    _gradcheck_real_imag(
        gradcheck_fn,
        func,
        func_out,
        tupled_inputs,
        outputs,
        eps,
        rtol,
        atol,
        check_grad_dtypes,
        check_backward_ad=check_backward_ad,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
    )

    # Short circuit because remaining tests rely on backward AD to be implemented
    if not check_backward_ad:
        return True

    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)

    if check_undefined_grad and check_backward_ad:
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    return True
