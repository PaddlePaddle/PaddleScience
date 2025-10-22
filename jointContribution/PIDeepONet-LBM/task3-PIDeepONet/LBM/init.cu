#include "common.h"
#include "lb.h"

void init() {
  int y, x;
  double ut, vt, rhot, uv;
  double G;

  dx = Ly / M;
  dt = dx;
  ci = dx / dt;
  rcc = 3.0 / ci / ci;
  tau_f = 0.51;
  nu = (tau_f - 0.5) * dt / 3.0;
  U0 = Re * nu / Ly;
  wf = 1.0 / tau_f;
  G = 8.0 * U0 * nu / Ly / Ly;
  rho_in = rho0 + 1.5 * Lx * G;
  rho_out = rho0 - 1.5 * Lx * G;
  Fx = 0.0;
  Fy = 0.0;

  sf[0] = sf[3] = sf[5] = 0.f;
  sf[7] = sf[8] = 1.f / tau_f;
  sf[4] = sf[6] = (16.f * tau_f - 8.f) / (8.f * tau_f - 1.f);
  sf[1] = 1.1;
  sf[2] = 1.2;

  printf("tau_f = %.3f, rho_in = %.3f, rho_out = %.3f, Ma = %.3f\n",
         tau_f,
         rho_in,
         rho_out,
         U0 / ci);

  for (y = 1; y < M1 + 1; y++)
    for (x = N16; x < N1 + N16; x++) {
      rhot = rho0;
      ut = 0.f;
      vt = 0.f;
      uv = ut * ut + vt * vt;
      f[0][y][x] = FEQ_0(rhot, ut, vt, uv);
      f[1][y][x] = FEQ_1(rhot, ut, vt, uv);
      f[2][y][x] = FEQ_2(rhot, ut, vt, uv);
      f[3][y][x] = FEQ_3(rhot, ut, vt, uv);
      f[4][y][x] = FEQ_4(rhot, ut, vt, uv);
      f[5][y][x] = FEQ_5(rhot, ut, vt, uv);
      f[6][y][x] = FEQ_6(rhot, ut, vt, uv);
      f[7][y][x] = FEQ_7(rhot, ut, vt, uv);
      f[8][y][x] = FEQ_8(rhot, ut, vt, uv);
    }
}
