# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import pytest
import sympy as sp

import ppsci


def test_multi_model_and_sdf():
    """Test for Vibration equation."""
    paddle.seed(2023)
    # construct symbolic equation
    x, y, z = sp.symbols("x y z")
    invars = (x, y, z)

    u = sp.Function("u")(*invars)
    v = sp.Function("v")(*invars)
    w = sp.Function("w")(*invars)
    p = sp.Function("p")(*invars)

    k = sp.Function("k")(u, v, w)
    ep = sp.Function("ep")(u, v, p)

    sdf = sp.Function("sdf")(*invars)
    sdf__x = sdf.diff(x)
    sdf__y = sdf.diff(y)
    sdf__z = sdf.diff(z)

    tmp1 = u * sdf + sdf__x * sdf__y - sdf__z
    tmp2 = ep * tmp1 * k
    out_var = tmp1 + tmp2

    model1 = ppsci.arch.MLP(
        (x.name, y.name, z.name), (u.name, v.name, w.name, p.name), 2, 8
    )
    model2 = ppsci.arch.MLP((u.name, v.name, w.name), (k.name,), 2, 6)
    model3 = ppsci.arch.MLP((u.name, v.name, p.name), (ep.name,), 2, 6)

    # translate symbolic equation to paddle function
    translated_func = ppsci.lambdify(
        out_var,
        (model1, model2, model3),
    )
    # prepare input dict
    geom = ppsci.geometry.Sphere([0, 0, 0], 2)
    input_dict = geom.sample_interior(
        100,
        compute_sdf_derivatives=True,
    )
    input_dict = {k: paddle.to_tensor(v) for k, v in input_dict.items()}
    input_dict_copy = {k: v for k, v in input_dict.items()}
    # compute out_var using translated function
    out_var_tensor = translated_func(input_dict)

    # compute out_var manually below
    uvwp = model1(input_dict_copy)
    u_eval, v_eval, w_eval, p_eval = (
        uvwp["u"],
        uvwp["v"],
        uvwp["w"],
        uvwp["p"],
    )
    k_eval = model2({**input_dict_copy, "u": u_eval, "v": v_eval, "w": w_eval})["k"]
    ep_eval = model3({**input_dict_copy, "u": u_eval, "v": v_eval, "p": p_eval})["ep"]
    sdf_eval = input_dict_copy["sdf"]
    sdf__x_eval = input_dict_copy["sdf__x"]
    sdf__y_eval = input_dict_copy["sdf__y"]
    sdf__z_eval = input_dict_copy["sdf__z"]

    tmp1_eval = u_eval * sdf_eval + sdf__x_eval * sdf__y_eval - sdf__z_eval
    tmp2_eval = ep_eval * tmp1_eval * k_eval
    out_var_reference = tmp1_eval + tmp2_eval

    assert np.allclose(out_var_tensor.numpy(), out_var_reference.numpy())


if __name__ == "__main__":
    pytest.main()
