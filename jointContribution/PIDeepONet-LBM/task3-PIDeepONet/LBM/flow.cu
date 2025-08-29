#include "common.h"
#include "lb.h"

__constant__ double diag[Q] = {1.f / 9,
                               1.f / 36,
                               1.f / 36,
                               1.f / 6,
                               1.f / 12,
                               1.0 / 6,
                               1.f / 12,
                               1.f / 4,
                               1.f / 4};
__constant__ int e_d[Q][2] = {{0, 0},

                              {1, 0},
                              {0, 1},
                              {-1, 0},
                              {0, -1},

                              {1, 1},
                              {-1, 1},
                              {-1, -1},
                              {1, -1}};

__constant__ int re_d[Q] = {0,
                            3,
                            4,
                            1,
                            2,

                            7,
                            8,
                            5,
                            6};

//--------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
__global__ void Evol_flow(
    double *s_d, double dt, double Fx, double Fy, double *f_d, double *F_d) {
  double RHO, U, V, UV;
  int tx, ty, k, x, y;
  double mf0, mf1, mf2, mf3, mf4, mf5, mf6, mf7, mf8;
  __shared__ double f[Q][BY][BX];

  tx = threadIdx.x;
  ty = threadIdx.y;
  x = N16 + blockIdx.x * BX + tx;
  y = (1 + blockIdx.y * BY + ty);
  k = NX2 * y + x;

  if (x < N16 + N1) {
    f[0][ty][tx] = f_d[k + 0 * NYNX2];
    f[1][ty][tx] = f_d[k + 1 * NYNX2];
    f[2][ty][tx] = f_d[k + 2 * NYNX2];
    f[3][ty][tx] = f_d[k + 3 * NYNX2];
    f[4][ty][tx] = f_d[k + 4 * NYNX2];
    f[5][ty][tx] = f_d[k + 5 * NYNX2];
    f[6][ty][tx] = f_d[k + 6 * NYNX2];
    f[7][ty][tx] = f_d[k + 7 * NYNX2];
    f[8][ty][tx] = f_d[k + 8 * NYNX2];

    // f-mf///////////////////////////
    mf0 = f[0][ty][tx] + f[1][ty][tx] + f[2][ty][tx] + f[3][ty][tx] +
          f[4][ty][tx] + f[5][ty][tx] + f[6][ty][tx] + f[7][ty][tx] +
          f[8][ty][tx];
    mf1 = -4 * f[0][ty][tx] - f[1][ty][tx] - f[2][ty][tx] - f[3][ty][tx] -
          f[4][ty][tx] +
          2 * (f[5][ty][tx] + f[6][ty][tx] + f[7][ty][tx] + f[8][ty][tx]);
    mf2 = 4 * f[0][ty][tx] -
          2 * (f[1][ty][tx] + f[2][ty][tx] + f[3][ty][tx] + f[4][ty][tx]) +
          f[5][ty][tx] + f[6][ty][tx] + f[7][ty][tx] + f[8][ty][tx];
    mf3 = f[1][ty][tx] - f[3][ty][tx] + f[5][ty][tx] - f[6][ty][tx] -
          f[7][ty][tx] + f[8][ty][tx];
    mf4 = -2 * (f[1][ty][tx] - f[3][ty][tx]) + f[5][ty][tx] - f[6][ty][tx] -
          f[7][ty][tx] + f[8][ty][tx];
    mf5 = f[2][ty][tx] - f[4][ty][tx] + f[5][ty][tx] + f[6][ty][tx] -
          f[7][ty][tx] - f[8][ty][tx];
    mf6 = -2 * (f[2][ty][tx] - f[4][ty][tx]) + f[5][ty][tx] + f[6][ty][tx] -
          f[7][ty][tx] - f[8][ty][tx];
    mf7 = f[1][ty][tx] - f[2][ty][tx] + f[3][ty][tx] - f[4][ty][tx];
    mf8 = f[5][ty][tx] - f[6][ty][tx] + f[7][ty][tx] - f[8][ty][tx];

    // macroscopic
    // variables/////////////////////////////////////////////////////////////////////////////
    RHO = RHO(ty, tx);
    U = UX(ty, tx) + 0.5 * dt * Fx;
    V = VY(ty, tx) + 0.5 * dt * Fy;
    UV = U * U + V * V;

    // collision//-------------------------------------------------------------------------------------------------------------------------------
    mf0 = (mf0 - s_d[0] * (mf0 - MEQ_0(RHO))) +
          dt * (1.f - 0.5 * s_d[0]) * F_0(U, V, Fx, Fy);
    mf1 = (mf1 - s_d[1] * (mf1 - MEQ_1(RHO, UV))) +
          dt * (1.f - 0.5 * s_d[1]) * F_1(U, V, Fx, Fy) * 6.f;
    mf2 = (mf2 - s_d[2] * (mf2 - MEQ_2(RHO, UV))) -
          dt * (1.f - 0.5 * s_d[2]) * F_2(U, V, Fx, Fy) * 6.f;
    mf3 = (mf3 - s_d[3] * (mf3 - MEQ_3(U))) +
          dt * (1.f - 0.5 * s_d[3]) * F_3(U, V, Fx, Fy);
    mf4 = (mf4 - s_d[4] * (mf4 - MEQ_4(U))) -
          dt * (1.f - 0.5 * s_d[4]) * F_4(U, V, Fx, Fy);
    mf5 = (mf5 - s_d[5] * (mf5 - MEQ_5(V))) +
          dt * (1.f - 0.5 * s_d[5]) * F_5(U, V, Fx, Fy);
    mf6 = (mf6 - s_d[6] * (mf6 - MEQ_6(V))) -
          dt * (1.f - 0.5 * s_d[6]) * F_6(U, V, Fx, Fy);
    mf7 = (mf7 - s_d[7] * (mf7 - MEQ_7(U, V))) +
          dt * (1.f - 0.5 * s_d[7]) * F_7(U, V, Fx, Fy) * 2.f;
    mf8 = (mf8 - s_d[8] * (mf8 - MEQ_8(U, V))) +
          dt * (1.f - 0.5 * s_d[8]) * F_8(U, V, Fx, Fy);

    //----------------------------------------------------------------------------------------------------------------------------------------
    mf0 = mf0 * diag[0];
    mf1 = mf1 * diag[1];
    mf2 = mf2 * diag[2];
    mf3 = mf3 * diag[3];
    mf4 = mf4 * diag[4];
    mf5 = mf5 * diag[5];
    mf6 = mf6 * diag[6];
    mf7 = mf7 * diag[7];
    mf8 = mf8 * diag[8];
    //--mf - f
    //--//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    f[0][ty][tx] = (mf0 - 4.f * (mf1 - mf2));
    f[1][ty][tx] = (mf0 - mf1 - 2.f * (mf2 + mf4) + mf3 + mf7);
    f[2][ty][tx] = (mf0 - mf1 - 2.f * (mf2 + mf6) + mf5 - mf7);
    f[3][ty][tx] = (mf0 - mf1 - 2.f * (mf2 - mf4) - mf3 + mf7);
    f[4][ty][tx] = (mf0 - mf1 - 2.f * (mf2 - mf6) - mf5 - mf7);
    f[5][ty][tx] = (mf0 + mf1 + mf1 + mf2 + mf3 + mf4 + mf5 + mf6 + mf8);
    f[6][ty][tx] = (mf0 + mf1 + mf1 + mf2 - mf3 - mf4 + mf5 + mf6 - mf8);
    f[7][ty][tx] = (mf0 + mf1 + mf1 + mf2 - mf3 - mf4 - mf5 - mf6 + mf8);
    f[8][ty][tx] = (mf0 + mf1 + mf1 + mf2 + mf3 + mf4 - mf5 - mf6 - mf8);

    __syncthreads();

    //  streaming
    //  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    F_d[k + 0 * NYNX2] = f[0][ty][tx];
    F_d[k + NX2 + 2 * NYNX2] = f[2][ty][tx];
    F_d[k - NX2 + 4 * NYNX2] = f[4][ty][tx];

    if (tx != 0) {
      F_d[k + 1 * NYNX2] = f[1][ty][tx - 1];
      F_d[k + NX2 + 5 * NYNX2] = f[5][ty][tx - 1];
      F_d[k - NX2 + 8 * NYNX2] = f[8][ty][tx - 1];
    }

    if (tx == BX - 1) {
      F_d[k + 1 + 1 * NYNX2] = f[1][ty][tx];
      F_d[k + NX2 + 1 + 5 * NYNX2] = f[5][ty][tx];
      F_d[k - NX2 + 1 + 8 * NYNX2] = f[8][ty][tx];
    }

    if (tx != BX - 1) {
      F_d[k + 3 * NYNX2] = f[3][ty][tx + 1];
      F_d[k + NX2 + 6 * NYNX2] = f[6][ty][tx + 1];
      F_d[k - NX2 + 7 * NYNX2] = f[7][ty][tx + 1];
    }

    if (tx == 0) {
      F_d[k - 1 + 3 * NYNX2] = f[3][ty][tx];
      F_d[k + NX2 - 1 + 6 * NYNX2] = f[6][ty][tx];
      F_d[k - NX2 - 1 + 7 * NYNX2] = f[7][ty][tx];
    }
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
/*
__global__ void Bc_flow_BB(int *flag_d, double *f_d)
{
  int tx, ty, k, x, y;
  int kq, xp, yp, kp;

  tx = threadIdx.x;           ty = threadIdx.y;
  x = N16+blockIdx.x*BX+tx;	y = (1+blockIdx.y*BY+ty);
  k = NX2*y+x;

  if (x < N16+N1)
  {
    if(flag_d[k] == 0)
    {
      for(kq = 1; kq < Q; kq++)
      {
        xp = x + e_d[kq][0]; yp = y + e_d[kq][1];
        kp = NX2*yp+xp;
        if(flag_d[kp] == 1)
        {
          f_d[kp + NYNX2*kq] = f_d[k + NYNX2*re_d[kq]];
        }

      }
    }
  }

}
*/

__global__ void Bc_flow_X(
    double dt, double Fx, double Fy, double U, double *f_d) {
  int x, y, k, k1;
  double f0, f1, f2, f3, f4, f5, f6, f7, f8;
  double vx1, vy1, vv1, vx, vy, vv, rho1, rho;

  ///////////////////////////////////////////////////////////////
  // boundary points 0/1 left/right
  y = 1 + blockIdx.x * NT + threadIdx.x;

  if (y <= M1) {
    if (blockIdx.y == 0) {
      x = N16;
      k = y * NX2 + x;
      k1 = k + 1;
    } else {
      x = N16 + N1 - 1;
      k = y * NX2 + x;
      k1 = k - 1;
    }

    f0 = f_d[k1 + 0 * NYNX2];
    f1 = f_d[k1 + 1 * NYNX2];
    f2 = f_d[k1 + 2 * NYNX2];
    f3 = f_d[k1 + 3 * NYNX2];
    f4 = f_d[k1 + 4 * NYNX2];
    f5 = f_d[k1 + 5 * NYNX2];
    f6 = f_d[k1 + 6 * NYNX2];
    f7 = f_d[k1 + 7 * NYNX2];
    f8 = f_d[k1 + 8 * NYNX2];

    vx1 = (f1 - f3 + f5 + f8 - f6 - f7) + 0.5 * dt * Fx;
    vy1 = (f2 - f4 + f5 + f6 - f7 - f8) + 0.5 * dt * Fy;
    vv1 = (vx1 * vx1 + vy1 * vy1);
    rho1 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    if (blockIdx.y == 0) {
      vx = 0.0;
      vy = 0.0;
      rho = rho1;
    } else {
      vx = 0.0;
      vy = 0.0;
      rho = rho1;
    }


    vv = (vx * vx + vy * vy);

    f_d[k + 0 * NYNX2] =
        FEQ_0(rho, vx, vy, vv) + f0 - FEQ_0(rho1, vx1, vy1, vv1);
    f_d[k + 1 * NYNX2] =
        FEQ_1(rho, vx, vy, vv) + f1 - FEQ_1(rho1, vx1, vy1, vv1);
    f_d[k + 2 * NYNX2] =
        FEQ_2(rho, vx, vy, vv) + f2 - FEQ_2(rho1, vx1, vy1, vv1);
    f_d[k + 3 * NYNX2] =
        FEQ_3(rho, vx, vy, vv) + f3 - FEQ_3(rho1, vx1, vy1, vv1);
    f_d[k + 4 * NYNX2] =
        FEQ_4(rho, vx, vy, vv) + f4 - FEQ_4(rho1, vx1, vy1, vv1);
    f_d[k + 5 * NYNX2] =
        FEQ_5(rho, vx, vy, vv) + f5 - FEQ_5(rho1, vx1, vy1, vv1);
    f_d[k + 6 * NYNX2] =
        FEQ_6(rho, vx, vy, vv) + f6 - FEQ_6(rho1, vx1, vy1, vv1);
    f_d[k + 7 * NYNX2] =
        FEQ_7(rho, vx, vy, vv) + f7 - FEQ_7(rho1, vx1, vy1, vv1);
    f_d[k + 8 * NYNX2] =
        FEQ_8(rho, vx, vy, vv) + f8 - FEQ_8(rho1, vx1, vy1, vv1);
  }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void Bc_flow_Y(
    double dt, double Fx, double Fy, double U, int n, double dn, double *f_d) {
  int x, y, k, k1;
  double f0, f1, f2, f3, f4, f5, f6, f7, f8;
  double vx1, vy1, vv1, rho1;
  double vx, vy, vv, rho;
  double xx;
  double dx = dt;
  double omega = 10.0;  // 1.0 + dn*n;
  double A = 0.5 * U;

  ///////////////////////////////////////////////////////////////
  // boundary points 0/1 upper/bottom
  x = N16 + blockIdx.x * NT + threadIdx.x;

  if (x < N16 + N1) {
    if (blockIdx.y == 0) {
      y = M1;
      k = y * NX2 + x;
      k1 = k - NX2;
    } else {
      y = 1;
      k = y * NX2 + x;
      k1 = k + NX2;
    }

    f0 = f_d[k1 + 0 * NYNX2];
    f1 = f_d[k1 + 1 * NYNX2];
    f2 = f_d[k1 + 2 * NYNX2];
    f3 = f_d[k1 + 3 * NYNX2];
    f4 = f_d[k1 + 4 * NYNX2];
    f5 = f_d[k1 + 5 * NYNX2];
    f6 = f_d[k1 + 6 * NYNX2];
    f7 = f_d[k1 + 7 * NYNX2];
    f8 = f_d[k1 + 8 * NYNX2];

    vx1 = (f1 - f3 + f5 + f8 - f6 - f7) + 0.5 * dt * Fx;
    vy1 = (f2 - f4 + f5 + f6 - f7 - f8) + 0.5 * dt * Fy;
    vv1 = (vx1 * vx1 + vy1 * vy1);
    rho1 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    if (blockIdx.y == 0) {
      xx = (x - N16) * dx;
      vx = U * (1. - cosh(10. * (xx - 0.5)) / cosh(5.)) +
           A * sin(2 * PI * xx) * sin(omega * n * dt);
      vy = 0.0;
      rho = rho1;
    } else {
      vx = 0.0;
      vy = 0.0;
      rho = rho1;
    }

    vv = (vx * vx + vy * vy);

    f_d[k + 0 * NYNX2] =
        FEQ_0(rho, vx, vy, vv) + f0 - FEQ_0(rho1, vx1, vy1, vv1);
    f_d[k + 1 * NYNX2] =
        FEQ_1(rho, vx, vy, vv) + f1 - FEQ_1(rho1, vx1, vy1, vv1);
    f_d[k + 2 * NYNX2] =
        FEQ_2(rho, vx, vy, vv) + f2 - FEQ_2(rho1, vx1, vy1, vv1);
    f_d[k + 3 * NYNX2] =
        FEQ_3(rho, vx, vy, vv) + f3 - FEQ_3(rho1, vx1, vy1, vv1);
    f_d[k + 4 * NYNX2] =
        FEQ_4(rho, vx, vy, vv) + f4 - FEQ_4(rho1, vx1, vy1, vv1);
    f_d[k + 5 * NYNX2] =
        FEQ_5(rho, vx, vy, vv) + f5 - FEQ_5(rho1, vx1, vy1, vv1);
    f_d[k + 6 * NYNX2] =
        FEQ_6(rho, vx, vy, vv) + f6 - FEQ_6(rho1, vx1, vy1, vv1);
    f_d[k + 7 * NYNX2] =
        FEQ_7(rho, vx, vy, vv) + f7 - FEQ_7(rho1, vx1, vy1, vv1);
    f_d[k + 8 * NYNX2] =
        FEQ_8(rho, vx, vy, vv) + f8 - FEQ_8(rho1, vx1, vy1, vv1);
  }
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
