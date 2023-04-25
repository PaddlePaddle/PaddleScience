"""Equations related to linear elasticity
"""

from typing import Optional

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class LinearElasticity(base.PDE):
    """Linear elasticity equations.
        Use either (E, nu) or (lambda_, mu) to define the material properties.

    Args:
        E (Optional[float]): The Young's modulus. Defaults to None.
        nu (Optional[float]): The Poisson's ratio. Defaults to None.
        lambda_ (Optional[float]): Lamé's first parameter. Defaults to None.
        mu (Optional[float], optional): Lamé's second parameter (shear modulus). Defaults to None. Defaults to None.
        rho (float, optional): Mass density.. Defaults to 1.
        dim (int, optional): Dimension of the linear elasticity (2 or 3). Defaults to 3.
        time (bool, optional): Whether contains time data. Defaults to False.
    """

    def __init__(
        self,
        E: Optional[float],
        nu: Optional[float],
        lambda_: Optional[float],
        mu: Optional[float],
        rho: float = 1,
        dim: int = 3,
        time: bool = False,
    ):
        super().__init__()
        self.E = E
        self.nu = nu
        self.lambda_ = lambda_
        self.mu = mu
        self.rho = rho
        self.dim = dim
        self.time = time

        # Stress equations
        def stress_disp_xx_compute_func(out):
            x, y, u, v = out["x"], out["y"], out["u"], out["v"]
            sigma_xx = out["sigma_xx"]
            stress_disp_xx = (
                self.lambda_ * (jacobian(u, x) + jacobian(v, y))
                + 2 * self.mu * jacobian(u, x)
                - sigma_xx
            )
            if self.dim == 3:
                z, w = out["z"], out["w"]
                stress_disp_xx += self.lambda_ * jacobian(w, z)
            return stress_disp_xx

        self.add_equation("stress_disp_xx", stress_disp_xx_compute_func)

        def stress_disp_yy_compute_func(out):
            x, y, u, v = out["x"], out["y"], out["u"], out["v"]
            sigma_yy = out["sigma_yy"]
            stress_disp_yy = (
                self.lambda_ * (jacobian(u, x) + jacobian(v, y))
                + 2 * self.mu * jacobian(v, y)
                - sigma_yy
            )
            if self.dim == 3:
                z, w = out["z"], out["w"]
                stress_disp_yy += self.lambda_ * jacobian(w, z)
            return stress_disp_yy

        self.add_equation("stress_disp_yy", stress_disp_yy_compute_func)

        if self.dim == 3:

            def stress_disp_zz_compute_func(out):
                x, y, z, u, v, w = (
                    out["x"],
                    out["y"],
                    out["z"],
                    out["u"],
                    out["v"],
                    out["w"],
                )
                sigma_zz = out["sigma_zz"]
                stress_disp_zz = (
                    self.lambda_ * (jacobian(u, x) + jacobian(v, y) + jacobian(w, z))
                    + 2 * self.mu * jacobian(w, z)
                    - sigma_zz
                )
                return stress_disp_zz

            self.add_equation("stress_disp_zz", stress_disp_zz_compute_func)

        def stress_disp_xy_compute_func(out):
            x, y, u, v = out["x"], out["y"], out["u"], out["v"]
            sigma_xy = out["sigma_xy"]
            stress_disp_xy = self.mu * (jacobian(u, y) + jacobian(v, x)) - sigma_xy
            return stress_disp_xy

        self.add_equation("stress_disp_xy", stress_disp_xy_compute_func)

        if self.dim == 3:

            def stress_disp_xz_compute_func(out):
                x, z, u, w = out["x"], out["z"], out["u"], out["w"]
                sigma_xz = out["sigma_xz"]
                stress_disp_xz = self.mu * (jacobian(u, z) + jacobian(w, x)) - sigma_xz
                return stress_disp_xz

            self.add_equation("stress_disp_xz", stress_disp_xz_compute_func)

            def stress_disp_yz_compute_func(out):
                y, z, v, w = out["y"], out["z"], out["v"], out["w"]
                sigma_yz = out["sigma_yz"]
                stress_disp_yz = self.mu * (jacobian(v, z) + jacobian(w, y)) - sigma_yz
                return stress_disp_yz

            self.add_equation("stress_disp_yz", stress_disp_yz_compute_func)

        # Equations of equilibrium
        def equilibrium_x_compute_func(out):
            x, y = out["x"], out["y"]
            sigma_xx, sigma_xy = out["sigma_xx"], out["sigma_xy"]
            equilibrium_x = -(jacobian(sigma_xx, x) + jacobian(sigma_xy, y))
            if self.dim == 3:
                z = out["z"]
                sigma_xz = out["sigma_xz"]
                equilibrium_x -= jacobian(sigma_xz, z)
            if self.time:
                t, u = out["t"], out["u"]
                equilibrium_x += self.rho * hessian(u, t)
            return equilibrium_x

        self.add_equation("equilibrium_x", equilibrium_x_compute_func)

        def equilibrium_y_compute_func(out):
            x, y = out["x"], out["y"]
            sigma_xy, sigma_yy = out["sigma_xy"], out["sigma_yy"]
            equilibrium_y = -(jacobian(sigma_xy, x) + jacobian(sigma_yy, y))
            if self.dim == 3:
                z = out["z"]
                sigma_yz = out["sigma_yz"]
                equilibrium_y -= jacobian(sigma_yz, z)
            if self.time:
                t, v = out["t"], out["v"]
                equilibrium_y += self.rho * hessian(v, t)
            return equilibrium_y

        self.add_equation("equilibrium_y", equilibrium_y_compute_func)

        if self.dim == 3:

            def equilibrium_z_compute_func(out):
                x, y, z = out["x"], out["y"], out["z"]
                sigma_xz, sigma_yz, sigma_zz = (
                    out["sigma_xz"],
                    out["sigma_yz"],
                    out["sigma_zz"],
                )
                equilibrium_z = -(
                    jacobian(sigma_xz, x)
                    + jacobian(sigma_yz, y)
                    + jacobian(sigma_zz, z)
                )
                if self.time:
                    t, w = out["t"], out["w"]
                    equilibrium_z += self.rho * hessian(w, t)
                return equilibrium_z

            self.add_equation("equilibrium_z", equilibrium_z_compute_func)

        # Traction equations
        def traction_x_compute_func(out):
            normal_x, normal_y = out["normal_x"], out["normal_y"]
            sigma_xx, sigma_xy = out["sigma_xx"], out["sigma_xy"]
            traction_x = normal_x * sigma_xx + normal_y * sigma_xy
            if self.dim == 3:
                normal_z = out["normal_z"]
                sigma_xz = out["sigma_xz"]
                traction_x += normal_z * sigma_xz
            return traction_x

        self.add_equation("traction_x", traction_x_compute_func)

        def traction_y_compute_func(out):
            normal_x, normal_y = out["normal_x"], out["normal_y"]
            sigma_xy, sigma_yy = out["sigma_xy"], out["sigma_yy"]
            traction_y = normal_x * sigma_xy + normal_y * sigma_yy
            if self.dim == 3:
                normal_z = out["normal_z"]
                sigma_yz = out["sigma_yz"]
                traction_y += normal_z * sigma_yz
            return traction_y

        self.add_equation("traction_y", traction_y_compute_func)

        def traction_z_compute_func(out):
            normal_x, normal_y, normal_z = (
                out["normal_x"],
                out["normal_y"],
                out["normal_z"],
            )
            sigma_xz, sigma_yz, sigma_zz = (
                out["sigma_xz"],
                out["sigma_yz"],
                out["sigma_zz"],
            )
            traction_z = normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz
            return traction_z

        self.add_equation("traction_z", traction_z_compute_func)

        # Navier equations
        def navier_x_compute_func(out):
            x, y, u, v = out["x"], out["y"], out["u"], out["v"]
            duxvywz = jacobian(u, x) + jacobian(v, y)
            duxxuyyuzz = hessian(u, x) + hessian(u, y)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                duxvywz += jacobian(w, z)
                duxxuyyuzz += hessian(u, z)
            navier_x = -(lambda_ + mu) * jacobian(duxvywz, x) - mu * duxxuyyuzz
            if self.time:
                t = out["t"]
                navier_x += rho * hessian(u, t)
            return navier_x

        self.add_equation("navier_x", navier_x_compute_func)

        def navier_y_compute_func(out):
            x, y, u, v = out["x"], out["y"], out["u"], out["v"]
            duxvywz = jacobian(u, x) + jacobian(v, y)
            dvxxvyyvzz = hessian(v, x) + hessian(v, y)
            if self.dim == 3:
                z, w = out["z"], out["w"]
                duxvywz += jacobian(w, z)
                dvxxvyyvzz += hessian(v, z)
            navier_y = -(lambda_ + mu) * jacobian(duxvywz, y) - mu * dvxxvyyvzz
            if self.time:
                t = out["t"]
                navier_y += rho * hessian(v, t)
            return navier_y

        self.add_equation("navier_y", navier_y_compute_func)

        if self.dim == 3:

            def navier_z_compute_func(out):
                x, y, z, u, v, w = (
                    out["x"],
                    out["y"],
                    out["z"],
                    out["u"],
                    out["v"],
                    out["w"],
                )
                duxvywz = jacobian(u, x) + jacobian(v, y) + jacobian(w, z)
                dwxxvyyvzz = hessian(w, x) + hessian(w, y) + hessian(w, z)
                navier_z = -(lambda_ + mu) * jacobian(duxvywz, z) - mu * dwxxvyyvzz
                if self.time:
                    t = out["t"]
                    navier_z += rho * hessian(w, t)
                return navier_z

            self.add_equation("navier_z", navier_z_compute_func)
