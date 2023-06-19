import paddle
import pytest
import sympy
from paddle import nn

from ppsci.equation import PDE


class Test_PDE:
    def test_pde_init(self):
        """
        Testing the PDE class initialization
        """
        pde = PDE()
        assert isinstance(pde, PDE)
        assert isinstance(pde.equations, dict)
        assert isinstance(pde.learnable_parameters, nn.ParameterList)

    def test_pde_add_equation(self):
        """
        initiate a PDE object and add an equation to it
        """
        pde = PDE()

        def simple_equation(out):
            x, y = out["x"], out["y"]
            return x + y

        pde.add_equation("simple", simple_equation)

        assert "simple" in pde.equations
        assert pde.equations["simple"] == simple_equation
        assert pde.equations["simple"]({"x": 1, "y": 2}) == 3

        # redefine the equation and add again
        def simple_equation2(out):
            x, y = out["x"], out["y"]
            return x - y

        pde.add_equation("simple", simple_equation2)

        assert pde.equations["simple"] == simple_equation2
        assert pde.equations["simple"]({"x": 1, "y": 2}) == -1

    def test_pde_create_symbols(self):
        """
        initiate a PDE object and add three symbols to it
        """
        pde = PDE()

        # create symbols
        x, y, z = pde.create_symbols("x y z")
        assert isinstance(x, sympy.Symbol)
        assert isinstance(y, sympy.Symbol)
        assert isinstance(z, sympy.Symbol)

    def test_pde_create_function(self):
        """
        initiate a PDE object and add a symbolic function to it
        """
        pde = PDE()

        # create symbols
        x, y, z = pde.create_symbols("x y z")

        # create a function
        f = pde.create_function(name="f", invars=(x, y, z))
        assert isinstance(f, sympy.Function)
        assert f.args == (x, y, z)

    def test_pde_parameters(self):
        """
        initiate a PDE object and add a learnable parameter to it
        """
        pde = PDE()

        assert len(pde.parameters()) == 0

        # add a learnable parameter
        pde.learnable_parameters.append(
            paddle.create_parameter(shape=[1], dtype="float32")
        )

        assert len(pde.parameters()) == 1

    def test_pde_state_dict(self):
        """
        initiate a PDE object, add a learnable parameter to it and check its state dict
        """
        pde = PDE()

        assert len(pde.state_dict()) == 0

        # add a learnable parameter
        pde.learnable_parameters.append(
            paddle.create_parameter(shape=[1], dtype="float32")
        )

        assert len(pde.state_dict()) == 1

    def test_pde_set_state_dict(self):
        """
        initiate a PDE object, set its state dict and check its state dict
        """
        pde = PDE()

        assert len(pde.state_dict()) == 0

        external_state = {
            "learnable_parameters": [
                paddle.create_parameter(shape=[1], dtype="float32")
            ]
        }

        # set state dict
        pde.set_state_dict(pde.state_dict())

        assert len(pde.state_dict()) == 1


if __name__ == "__main__":
    pytest.main()
