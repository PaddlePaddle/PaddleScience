import numpy as np


def thickness_dist(t, x, CTE=True):
    if CTE:
        a = -0.1036
    else:
        a = -0.1015
    return 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 
        0.2843 * x ** 3 + a * x ** 4)


def camber_line(params, x):
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)
    if len(params) == 2:
        m = params[0] / 100
        p = params[1] / 10
        if p == 0:
            dy_c = -2 * m * x
            return y_c, dy_c
        elif p == 1:
            dy_c = 2 * m * (1 - x)
            return y_c, dy_c
        mask1 = x < p
        mask2 = x >= p
        y_c[mask1] = m / p ** 2 * (2 * p * x[mask1] - x[mask1] ** 2)
        dy_c[mask1] = 2 * m / p ** 2 * (p - x[mask1])
        y_c[mask2] = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x[mask2] - x[
            mask2] ** 2)
        dy_c[mask2] = 2 * m / (1 - p) ** 2 * (p - x[mask2])
    elif len(params) == 3:
        l, p, q = params
        c_l, x_f = 3 / 20 * l, p / 20
        f = lambda x: x * (1 - np.sqrt(x / 3)) - x_f
        df = lambda x: 1 - 3 * np.sqrt(x / 3) / 2
        old_m = 0.5
        cond = True
        while cond:
            new_m = np.max([old_m - f(old_m) / df(old_m), 0])
            cond = np.abs(old_m - new_m) > 1e-15
            old_m = new_m
        m = old_m
        r = (3 * m - 7 * m ** 2 + 8 * m ** 3 - 4 * m ** 4) / np.sqrt(m * (1 -
            m)) - 3 / 2 * (1 - 2 * m) * (np.pi / 2 - np.arcsin(1 - 2 * m))
        k_1 = c_l / r
        mask1 = x <= m
        mask2 = x > m
        if q == 0:
            y_c[mask1] = k_1 * (x[mask1] ** 3 - 3 * m * x[mask1] ** 2 + m **
                2 * (3 - m) * x[mask1])
            dy_c[mask1] = k_1 * (3 * x[mask1] ** 2 - 6 * m * x[mask1] + m **
                2 * (3 - m))
            y_c[mask2] = k_1 * m ** 3 * (1 - x[mask2])
            dy_c[mask2] = -k_1 * m ** 3 * np.ones_like(dy_c[mask2])
        elif q == 1:
            k = (3 * (m - x_f) ** 2 - m ** 3) / (1 - m) ** 3
            y_c[mask1] = k_1 * ((x[mask1] - m) ** 3 - k * (1 - m) ** 3 * x[
                mask1] - m ** 3 * x[mask1] + m ** 3)
            dy_c[mask1] = k_1 * (3 * (x[mask1] - m) ** 2 - k * (1 - m) ** 3 -
                m ** 3)
            y_c[mask2] = k_1 * (k * (x[mask2] - m) ** 3 - k * (1 - m) ** 3 *
                x[mask2] - m ** 3 * x[mask2] + m ** 3)
            dy_c[mask2] = k_1 * (3 * k * (x[mask2] - m) ** 2 - k * (1 - m) **
                3 - m ** 3)
        else:
            raise ValueError(
                'Q must be 0 for normal camber or 1 for reflex camber.')
    else:
        raise ValueError(
            'The first input must be a tuple of the 2 or 3 digits that represent the camber line.'
            )
    return y_c, dy_c


def naca_generator(params, nb_samples=400, scale=1, origin=(0, 0),
    cosine_spacing=True, verbose=True, CTE=True):
    if len(params) == 3:
        params_c = params[:2]
        t = params[2] / 100
        if verbose:
            print(
                f'Generating naca M = {params_c[0]}, P = {params_c[1]}, XX = {t * 100}'
                )
    elif len(params) == 4:
        params_c = params[:3]
        t = params[3] / 100
        if verbose:
            print(
                f'Generating naca L = {params_c[0]}, P = {params_c[1]}, Q = {params_c[2]}, XX = {t * 100}'
                )
    else:
        raise ValueError(
            'The first argument must be a tuple of the 4 or 5 digits of the airfoil.'
            )
    if cosine_spacing:
        beta = np.pi * np.linspace(1, 0, nb_samples + 1, endpoint=True)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(1, 0, nb_samples + 1, endpoint=True)
    y_c, dy_c = camber_line(params_c, x)
    y_t = thickness_dist(t, x, CTE)
    theta = np.arctan(dy_c)
    x_u = x - y_t * np.sin(theta)
    x_l = x + y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    y_l = y_c - y_t * np.cos(theta)
    x = np.concatenate([x_u, x_l[:-1][::-1]], axis=0)
    y = np.concatenate([y_u, y_l[:-1][::-1]], axis=0)
    pos = np.stack([x * scale + origin[0], y * scale + origin[1]], axis=-1)
    pos[0], pos[-1] = np.array([1, 0]), np.array([1, 0])
    return pos
