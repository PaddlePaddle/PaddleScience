"""simple Sympy helper functions
"""

from symengine import sqrt


def line(x, point_x_1, point_y_1, point_x_2, point_y_2):
    """
    line function from point intercepts

    Parameters
    ----------
    x : Sympy Symbol/Exp
      the `x` in equation `y=a*x+b`
    point_x_1 : Sympy Symbol/Exp, float, int
      first intercept x position
    point_y_1 : Sympy Symbol/Exp, float, int
      first intercept y position
    point_x_2 : Sympy Symbol/Exp, float, int
      second intercept x position
    point_y_2 : Sympy Symbol/Exp, float, int
      second intercept y position

    Returns
    -------
    y : Sympy Expr
      `y=slope*x+intercept`
    """

    slope = (point_y_1 - point_y_2) / (point_x_1 - point_x_2)
    intercept = point_y_1 - slope * point_x_1
    return slope * x + intercept


def parabola(x, inter_1, inter_2, height):
    """
    parabola from point intercepts

    Parameters
    ----------
    x : Sympy Symbol/Exp
      the `x` in equation `y=a*x*2+b*x+c`
    inter_1 : Sympy Symbol/Exp, float, int
      first intercept such that `y=0` when `x=inter_1`
    inter_2 : Sympy Symbol/Exp, float, int
      second intercept such that `y=0` when `x=inter_1`
    height : Sympy Symbol/Exp, float, int
      max height of parabola

    Returns
    -------
    y : Sympy Expr
      `y=factor*(x-inter_1)*(x-+inter_2)`
    """

    factor = (4 * height) / (-(inter_1 ** 2) - inter_2 ** 2 + 2 * inter_1 * inter_2)
    return factor * (x - inter_1) * (x - inter_2)


def parabola2D(x, y, inter_1_x, inter_2_x, inter_1_y, inter_2_y, height):
    """
    square parabola from point intercepts

    Parameters
    ----------
    x : Sympy Symbol/Exp
      the `x` in equation `z=parabola(x)*parabola(y)`
    y : Sympy Symbol/Exp
      the `y` in equation `z=a*x**2+b*y**2+c*xy+d*y+e*x+f`
    inter_1_x : Sympy Symbol/Exp, float, int
      first intercept such that `z=0` when `x=inter_1_x`
    inter_2_x : Sympy Symbol/Exp, float, int
      second intercept such that `z=0` when `x=inter_2_x`
    inter_1_y : Sympy Symbol/Exp, float, int
      first intercept such that `z=0` when `y=inter_1_y`
    inter_2_y : Sympy Symbol/Exp, float, int
      second intercept such that `z=0` when `y=inter_2_y`
    height : Sympy Symbol/Exp, float, int
      max height of parabola

    Returns
    -------
    y : Sympy Expr
      `y=factor*(x-inter_1)*(x-+inter_2)`
    """

    parabola_x = parabola(x, inter_1_x, inter_2_x, sqrt(height))
    parabola_y = parabola(y, inter_1_y, inter_2_y, sqrt(height))
    return parabola_x * parabola_y
