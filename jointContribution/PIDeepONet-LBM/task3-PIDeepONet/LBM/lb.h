#ifndef __LB_H_
#define __LB_H_

#define FEQ_0(rho, u, v, uv) ((4.0 / 9.0 * (rho - 1.5 * uv)))
#define FEQ_1(rho, u, v, uv) \
  ((1.0 / 9.0 * (rho + 3.0 * u + 4.5 * u * u - 1.5 * uv)))
#define FEQ_2(rho, u, v, uv) \
  ((1.0 / 9.0 * (rho + 3.0 * v + 4.5 * v * v - 1.5 * uv)))
#define FEQ_3(rho, u, v, uv) \
  ((1.0 / 9.0 * (rho - 3.0 * u + 4.5 * u * u - 1.5 * uv)))
#define FEQ_4(rho, u, v, uv) \
  ((1.0 / 9.0 * (rho - 3.0 * v + 4.5 * v * v - 1.5 * uv)))
#define FEQ_5(rho, u, v, uv) \
  ((1.0 / 36.0 * (rho + 3.0 * (u + v) + 4.5 * (u + v) * (u + v) - 1.5 * uv)))
#define FEQ_6(rho, u, v, uv) \
  ((1.0 / 36.0 * (rho + 3.0 * (-u + v) + 4.5 * (-u + v) * (-u + v) - 1.5 * uv)))
#define FEQ_7(rho, u, v, uv) \
  ((1.0 / 36.0 * (rho + 3.0 * (-u - v) + 4.5 * (-u - v) * (-u - v) - 1.5 * uv)))
#define FEQ_8(rho, u, v, uv) \
  ((1.0 / 36.0 * (rho + 3.0 * (u - v) + 4.5 * (u - v) * (u - v) - 1.5 * uv)))
//-------------------------------------------------------------------------------------------------------------------------
#define MEQ_0(rho) (rho)
#define MEQ_1(rho, uv) (-2.0f * rho + 3.0f * uv)
#define MEQ_2(rho, uv) (rho - 3.0f * uv)
#define MEQ_3(u) (u)
#define MEQ_4(u) (-u)
#define MEQ_5(v) (v)
#define MEQ_6(v) (-v)
#define MEQ_7(u, v) (u * u - v * v)
#define MEQ_8(u, v) (u * v)
//--------------------------------------------------------------------------------------------------------------------------
#define F_0(u, v, Fx, Fy) (0.f)
#define F_1(u, v, Fx, Fy) (u * Fx + v * Fy)
#define F_2(u, v, Fx, Fy) (u * Fx + v * Fy)
#define F_3(u, v, Fx, Fy) (Fx)
#define F_4(u, v, Fx, Fy) (Fx)
#define F_5(u, v, Fx, Fy) (Fy)
#define F_6(u, v, Fx, Fy) (Fy)
#define F_7(u, v, Fx, Fy) (u * Fx - v * Fy)
#define F_8(u, v, Fx, Fy) (u * Fy + v * Fx)
//--------------------------------------------------------------------------------------------------------------------------
#define GEQ_0(C, u, v, uv) ((4.0 / 9.0 * C * (1.f - 1.5 * uv)))
#define GEQ_1(C, u, v, uv) \
  ((1.0 / 9.0 * C * (1.f + 3.0 * u + 4.5 * u * u - 1.5 * uv)))
#define GEQ_2(C, u, v, uv) \
  ((1.0 / 9.0 * C * (1.f + 3.0 * v + 4.5 * v * v - 1.5 * uv)))
#define GEQ_3(C, u, v, uv) \
  ((1.0 / 9.0 * C * (1.f - 3.0 * u + 4.5 * u * u - 1.5 * uv)))
#define GEQ_4(C, u, v, uv) \
  ((1.0 / 9.0 * C * (1.f - 3.0 * v + 4.5 * v * v - 1.5 * uv)))
#define GEQ_5(C, u, v, uv) \
  ((1.0 / 36.0 * C *       \
    (1.f + 3.0 * (u + v) + 4.5 * (u + v) * (u + v) - 1.5 * uv)))
#define GEQ_6(C, u, v, uv) \
  ((1.0 / 36.0 * C *       \
    (1.f + 3.0 * (-u + v) + 4.5 * (-u + v) * (-u + v) - 1.5 * uv)))
#define GEQ_7(C, u, v, uv) \
  ((1.0 / 36.0 * C *       \
    (1.f + 3.0 * (-u - v) + 4.5 * (-u - v) * (-u - v) - 1.5 * uv)))
#define GEQ_8(C, u, v, uv) \
  ((1.0 / 36.0 * C *       \
    (1.f + 3.0 * (u - v) + 4.5 * (u - v) * (u - v) - 1.5 * uv)))
//--------------------------------------------------------------------------------------------------------------------------
#define MGEQ_0(C) (C)
#define MGEQ_1(C, uv) C *(-2.0f + 3.0f * uv)
#define MGEQ_2(C, uv) C *(1.f - 3.0f * uv)
#define MGEQ_3(C, u) C *(u)
#define MGEQ_4(C, u) C *(-u)
#define MGEQ_5(C, v) C *(v)
#define MGEQ_6(C, v) C *(-v)
#define MGEQ_7(C, u, v) C *(u * u - v * v)
#define MGEQ_8(C, u, v) C *(u * v)
//--------------------------------------------------------------------------------------------------------------------------
#define RHO(y, x)                                                   \
  (f[0][y][x] + f[1][y][x] + f[2][y][x] + f[3][y][x] + f[4][y][x] + \
   f[5][y][x] + f[6][y][x] + f[7][y][x] + f[8][y][x])
#define UX(y, x) \
  (f[1][y][x] + f[5][y][x] + f[8][y][x] - f[3][y][x] - f[6][y][x] - f[7][y][x])
#define VY(y, x) \
  (f[2][y][x] + f[5][y][x] + f[6][y][x] - f[4][y][x] - f[7][y][x] - f[8][y][x])
#define C(y, x)                                                     \
  (g[0][y][x] + g[1][y][x] + g[2][y][x] + g[3][y][x] + g[4][y][x] + \
   g[5][y][x] + g[6][y][x] + g[7][y][x] + g[8][y][x])

//--------------------------------------------------------------------------------------------------------------------------
void geo();
void init();
__global__ void Evol_flow(
    double *s_d, double dt, double Fx, double Fy, double *f_d, double *F_d);
__global__ void Bc_flow_X(
    double dt, double Fx, double Fy, double U, double *f_d);
__global__ void Bc_flow_Y(double dt, double Fx, double Fy, double *f_d);
__global__ void Bc_flow_BB(int *flag_d, double *f_d);
__global__ void Evol_solute(double *s_d,
                            double dt,
                            double Fx,
                            double Fy,
                            double *g_d,
                            double *G_d,
                            double *f_d);
__global__ void Bc_solute_X(
    double dt, double Fx, double Fy, double *g_d, double *f_d);
__global__ void Bc_solute_Y(
    double dt, double Fx, double Fy, double *g_d, double *f_d);
__global__ void Bc_solute_BB();
double error();
void datasave();

#endif
