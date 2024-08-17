#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include<time.h>

#define M 1024      // grid in y-direction
#define N 256       // grid in x-direction
#define M1 (M + 1)  // number of grid in y-direction
#define N1 (N + 1)  // number of grid in x-direction

#define At 0.1
#define rhol 1.0
#define rhog (rhol * (1 - At) / (1 + At))
#define rhom (0.5 * (rhol + rhog))

#define phil 1.0
#define phig (-1.0)
#define phim (0.5 * (phil + phig))

#define D 4.0
//#define sigma 0.0000526
#define sigma (5.0e-5)

#define pi 3.1415926535897932
#define Max 80000

#define Q 9  // 9 velocities in LBM

#define BX 128  //每一个block的大小
#define BY 1

#define dx 1.0  // c=1.0
#define dt 1.0
#define rdt 1.0

const int N16 = 16;
const int NX = (N1 + N16 + N16) / 16 * 16;  //带16层拓展   N16到N16+N1-1
const int NY = M1 + 2;                      // 带1层拓展   1到M1
const int NYNX = NY * NX;

void init();                                             //初始化
void datadeal(int step);                                 //输出函数
double error1(double phi[NY][NX], double phi0[NY][NX]);  //计算组分f的密度误差
// double A_spike( );
// double A_bulble( );

double f[Q][NY][NX] = {0.0},
       g[Q][NY][NX] = {0.0};  //分配CPU分布函数 f(液) g(气)的空间
double phi[NY][NX] = {0.0}, rho[NY][NX] = {0.0}, p[NY][NX] = {0.0},
       mu[NY][NX] = {0.0};
double u[NY][NX] = {0.0}, v[NY][NX] = {0.0};
double phi0[NY][NX] = {0.0}, u0[NY][NX] = {0.0}, v0[NY][NX] = {0.0};

int e[Q][2], w[Q];

double Re;
double ww_f, ww_g;  // w=1/tau
double beta, Kappa, MM, A, ggy;

//////////////////////////////////////////////////////////
#define FEQ_0(phi, vx, vy, mu, A) \
  (phi - 5.0 / 9.0 * A * mu)  //速度vx,vy都没有除以c, vv=1.5*(vx*vx+vy*vy)
#define FEQ_1(phi, vx, vy, mu, A) (1.0 / 9.0 * (A * mu + 3.0 * phi * vx))
#define FEQ_2(phi, vx, vy, mu, A) (1.0 / 9.0 * (A * mu + 3.0 * phi * vy))
#define FEQ_3(phi, vx, vy, mu, A) (1.0 / 9.0 * (A * mu - 3.0 * phi * vx))
#define FEQ_4(phi, vx, vy, mu, A) (1.0 / 9.0 * (A * mu - 3.0 * phi * vy))
#define FEQ_5(phi, vx, vy, mu, A) \
  (1.0 / 36.0 * (A * mu + 3.0 * phi * (vx + vy)))
#define FEQ_6(phi, vx, vy, mu, A) \
  (1.0 / 36.0 * (A * mu + 3.0 * phi * (-vx + vy)))
#define FEQ_7(phi, vx, vy, mu, A) \
  (1.0 / 36.0 * (A * mu + 3.0 * phi * (-vx - vy)))
#define FEQ_8(phi, vx, vy, mu, A) \
  (1.0 / 36.0 * (A * mu + 3.0 * phi * (vx - vy)))
//////////////////////////////////////////////////////////
#define GEQ_0(rho, p, vx, vy, vv)     \
  (4.0 / 9.0 * (3.0 * p - rho * vv) - \
   3 * p)  //速度vx,vy都没有除以c, vv=1.5*(vx*vx+vy*vy)
#define GEQ_1(rho, p, vx, vy, vv) \
  (1.0 / 9.0 * (3.0 * p + rho * (3.0 * vx + 4.5 * vx * vx - vv)))
#define GEQ_2(rho, p, vx, vy, vv) \
  (1.0 / 9.0 * (3.0 * p + rho * (3.0 * vy + 4.5 * vy * vy - vv)))
#define GEQ_3(rho, p, vx, vy, vv) \
  (1.0 / 9.0 * (3.0 * p + rho * (-3.0 * vx + 4.5 * vx * vx - vv)))
#define GEQ_4(rho, p, vx, vy, vv) \
  (1.0 / 9.0 * (3.0 * p + rho * (-3.0 * vy + 4.5 * vy * vy - vv)))
#define GEQ_5(rho, p, vx, vy, vv) \
  (1.0 / 36.0 *                   \
   (3.0 * p + rho * (3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - vv)))
#define GEQ_6(rho, p, vx, vy, vv) \
  (1.0 / 36.0 *                   \
   (3.0 * p + rho * (3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - vv)))
#define GEQ_7(rho, p, vx, vy, vv) \
  (1.0 / 36.0 *                   \
   (3.0 * p + rho * (3.0 * (-vx - vy) + 4.5 * (-vx - vy) * (-vx - vy) - vv)))
#define GEQ_8(rho, p, vx, vy, vv) \
  (1.0 / 36.0 *                   \
   (3.0 * p + rho * (3.0 * (vx - vy) + 4.5 * (vx - vy) * (vx - vy) - vv)))
//////////////////////////////////////////////////////////
#define WEQ_0(vx, vy, vv) (4.0 / 9.0 * (1.0 - vv))
#define WEQ_1(vx, vy, vv) (1.0 / 9.0 * (1.0 + 3.0 * vx + 4.5 * vx * vx - vv))
#define WEQ_2(vx, vy, vv) (1.0 / 9.0 * (1.0 + 3.0 * vy + 4.5 * vy * vy - vv))
#define WEQ_3(vx, vy, vv) (1.0 / 9.0 * (1.0 - 3.0 * vx + 4.5 * vx * vx - vv))
#define WEQ_4(vx, vy, vv) (1.0 / 9.0 * (1.0 - 3.0 * vy + 4.5 * vy * vy - vv))
#define WEQ_5(vx, vy, vv) \
  (1.0 / 36.0 * (1.0 + 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - vv))
#define WEQ_6(vx, vy, vv) \
  (1.0 / 36.0 * (1.0 + 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - vv))
#define WEQ_7(vx, vy, vv) \
  (1.0 / 36.0 * (1.0 + 3.0 * (-vx - vy) + 4.5 * (-vx - vy) * (-vx - vy) - vv))
#define WEQ_8(vx, vy, vv) \
  (1.0 / 36.0 * (1.0 + 3.0 * (vx - vy) + 4.5 * (vx - vy) * (vx - vy) - vv))
//////////////////////////////////////////////////////////
#define MFEQ_0(phi, vx, vy, mu, A) (phi)
#define MFEQ_1(phi, vx, vy, mu, A) (-4.0 * phi + 2.0 * A * mu)
#define MFEQ_2(phi, vx, vy, mu, A) (4.0 * phi - 3.0 * A * mu)
#define MFEQ_3(phi, vx, vy, mu, A) (phi * vx)
#define MFEQ_4(phi, vx, vy, mu, A) (-phi * vx)
#define MFEQ_5(phi, vx, vy, mu, A) (phi * vy)
#define MFEQ_6(phi, vx, vy, mu, A) (-phi * vy)
#define MFEQ_7(phi, vx, vy, mu, A) (0.)
#define MFEQ_8(phi, vx, vy, mu, A) (0.)
//////////////////////////////////////////////////////////
#define MGEQ_0(rho, p, vx, vy) \
  (0.)  //速度vx,vy都没有除以c, vv=1.5*(vx*vx+vy*vy)
#define MGEQ_1(rho, p, vx, vy) (6.0 * p + 3.0 * rho * (vx * vx + vy * vy))
#define MGEQ_2(rho, p, vx, vy) (-9.0 * p - 3.0 * rho * (vx * vx + vy * vy))
#define MGEQ_3(rho, p, vx, vy) (rho * vx)
#define MGEQ_4(rho, p, vx, vy) (-rho * vx)
#define MGEQ_5(rho, p, vx, vy) (rho * vy)
#define MGEQ_6(rho, p, vx, vy) (-rho * vy)
#define MGEQ_7(rho, p, vx, vy) (rho * (vx * vx - vy * vy))
#define MGEQ_8(rho, p, vx, vy) (rho * vx * vy)

//////////////////////////////////////////////////////////
#define MFF_0(phi, vx, vy, phi0, vx0, vy0) \
  (0.)  //速度vx,vy都没有除以c, vv=1.5*(vx*vx+vy*vy)
#define MFF_1(phi, vx, vy, phi0, vx0, vy0) (0.)
#define MFF_2(phi, vx, vy, phi0, vx0, vy0) (0.)
#define MFF_3(phi, vx, vy, phi0, vx0, vy0) (phi * vx - phi0 * vx0)
#define MFF_4(phi, vx, vy, phi0, vx0, vy0) (-phi * vx + phi0 * vx0)
#define MFF_5(phi, vx, vy, phi0, vx0, vy0) (phi * vy - phi0 * vy0)
#define MFF_6(phi, vx, vy, phi0, vx0, vy0) (-phi * vy + phi0 * vy0)
#define MFF_7(phi, vx, vy, phi0, vx0, vy0) (0.)
#define MFF_8(phi, vx, vy, phi0, vx0, vy0) (0.)
//////////////////////////////////////////////////////////
#define diag0 (1.0 / 9.0)
#define diag1 (1.0 / 36.0)
#define diag2 (1.0 / 36.0)
#define diag3 (1.0 / 6.0)
#define diag4 (1.0 / 12.0)
#define diag5 (1.0 / 6.0)
#define diag6 (1.0 / 12.0)
#define diag7 (1.0 / 4.0)
#define diag8 (1.0 / 4.0)
////////////////////////////////////////////////////////////////////////////
#define F_d(k, y, x) \
  F_d[(k)*NYNX + (y)*NX + (x)]  //分布函数F在y行x列k方向的offset
#define G_d(k, y, x) \
  G_d[(k)*NYNX + (y)*NX + (x)]  //分布函数F在y行x列k方向的offset
#define phi_d(y, x) phi_d[(y)*NX + (x)]
#define rho_d(y, x) rho_d[(y)*NX + (x)]
#define mu_d(y, x) mu_d[(y)*NX + (x)]
#define p_d(y, x) p_d[(y)*NX + (x)]
#define u_d(y, x) u_d[(y)*NX + (x)]
#define v_d(y, x) v_d[(y)*NX + (x)]
#define u0_d(y, x) u0_d[(y)*NX + (x)]
#define v0_d(y, x) v0_d[(y)*NX + (x)]
#define phi0_d(y, x) phi0_d[(y)*NX + (x)]

///////////////////////////////////////////////////////////////////////////////////
double *f_dev, *F_dev, *g_dev, *G_dev;  //给分布函数f,g分配GPU中内存
double *phi_dev, *rho_dev, *p_dev, *phi0_dev, *mu_dev;
double *u_dev, *v_dev, *u0_dev, *v0_dev;
///////////////////////////////////////////////////////////////////////////////////
__global__ void collision_propagation(double A,
                                      double MM,
                                      double w_f,
                                      double w_g,
                                      double ggy,
                                      double *phi_d,
                                      double *mu_d,
                                      double *rho_d,
                                      double *p_d,
                                      double *u_d,
                                      double *v_d,
                                      double *phi0_d,
                                      double *u0_d,
                                      double *v0_d,
                                      double *f_d,
                                      double *F_d,
                                      double *g_d,
                                      double *G_d);
__global__ void Macro_rho(double *phi_d,
                          double *phi0_d,
                          double *rho_d,
                          double *f_d);
__global__ void Macro_mu(double Kappa,
                         double beta,
                         double *phi_d,
                         double *mu_d);
__global__ void Macro_u(double ggy,
                        double MM,
                        double *phi_d,
                        double *rho_d,
                        double *p_d,
                        double *mu_d,
                        double *u_d,
                        double *v_d,
                        double *u0_d,
                        double *v0_d,
                        double *g_d);

////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int m, k;
  double err1;

  e[0][0] = e[0][1] = 0;
  e[1][0] = 1;
  e[1][1] = 0;
  e[2][0] = 0;
  e[2][1] = 1;
  e[3][0] = -1;
  e[3][1] = 0;
  e[4][0] = 0;
  e[4][1] = -1;
  e[5][0] = 1;
  e[5][1] = 1;
  e[6][0] = -1;
  e[6][1] = 1;
  e[7][0] = -1;
  e[7][1] = -1;
  e[8][0] = 1;
  e[8][1] = -1;

  w[0] = 4.0 / 9.0;
  w[1] = w[2] = w[3] = w[4] = 1.0 / 9.0;
  w[5] = w[6] = w[7] = w[8] = 1.0 / 36.0;

  for (k = 0; k < 50; k++) {
    Re = 20.0 + k * 20.;

    init();  //初始化

    err1 = 1.0;
    m = 0;

    datadeal(m);  //输出初始的u,v,rho

    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    printf("Simulation running on %s\n", properties.name);

    ////////////////////////////////////////////
    dim3 threads(BX, BY);  //每个block的维数大小定义
    dim3 grid((N1 + BX - 1) / BX, (M1 + BY - 1) / BY);  // grid维数大小定义

    // GPU中显存分配: f_dev[], F_dev[]
    cudaMalloc((void **)&f_dev, sizeof(double) * Q * NY * NX);
    cudaMalloc((void **)&F_dev, sizeof(double) * Q * NY * NX);
    cudaMalloc((void **)&g_dev, sizeof(double) * Q * NY * NX);
    cudaMalloc((void **)&G_dev, sizeof(double) * Q * NY * NX);
    cudaMalloc((void **)&phi_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&rho_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&p_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&phi0_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&mu_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&u_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&v_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&u0_dev, sizeof(double) * NY * NX);
    cudaMalloc((void **)&v0_dev, sizeof(double) * NY * NX);

    // 复制数据(GPU <= CPU): f_dev <= f
    cudaMemcpy(f_dev,
               &f[0][0][0],
               sizeof(double) * Q * NY * NX,
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_dev,
               &g[0][0][0],
               sizeof(double) * Q * NY * NX,
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        phi_dev, &phi[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        rho_dev, &rho[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        p_dev, &p[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(phi0_dev,
               &phi0[0][0],
               sizeof(double) * NY * NX,
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        mu_dev, &mu[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        u_dev, &u[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        v_dev, &v[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        u0_dev, &u0[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(
        v0_dev, &v0[0][0], sizeof(double) * NY * NX, cudaMemcpyHostToDevice);


    // time_begin=clock();
    while (m < Max) {
      collision_propagation<<<grid, threads>>>(A,
                                               MM,
                                               ww_f,
                                               ww_g,
                                               ggy,
                                               phi_dev,
                                               mu_dev,
                                               rho_dev,
                                               p_dev,
                                               u_dev,
                                               v_dev,
                                               phi0_dev,
                                               u0_dev,
                                               v0_dev,
                                               f_dev,
                                               F_dev,
                                               g_dev,
                                               G_dev);
      Macro_rho<<<grid, threads>>>(phi_dev, phi0_dev, rho_dev, F_dev);
      Macro_mu<<<grid, threads>>>(Kappa, beta, phi_dev, mu_dev);
      Macro_u<<<grid, threads>>>(ggy,
                                 MM,
                                 phi_dev,
                                 rho_dev,
                                 p_dev,
                                 mu_dev,
                                 u_dev,
                                 v_dev,
                                 u0_dev,
                                 v0_dev,
                                 G_dev);

      collision_propagation<<<grid, threads>>>(A,
                                               MM,
                                               ww_f,
                                               ww_g,
                                               ggy,
                                               phi_dev,
                                               mu_dev,
                                               rho_dev,
                                               p_dev,
                                               u_dev,
                                               v_dev,
                                               phi0_dev,
                                               u0_dev,
                                               v0_dev,
                                               F_dev,
                                               f_dev,
                                               G_dev,
                                               g_dev);
      Macro_rho<<<grid, threads>>>(phi_dev, phi0_dev, rho_dev, f_dev);
      Macro_mu<<<grid, threads>>>(Kappa, beta, phi_dev, mu_dev);
      Macro_u<<<grid, threads>>>(ggy,
                                 MM,
                                 phi_dev,
                                 rho_dev,
                                 p_dev,
                                 mu_dev,
                                 u_dev,
                                 v_dev,
                                 u0_dev,
                                 v0_dev,
                                 g_dev);

      m = m + 2;

      if (m % 10000 == 0) {
        cudaMemcpy(&f[0][0][0],
                   f_dev,
                   Q * NY * NX * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(
            &u[0][0], u_dev, NY * NX * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            &v[0][0], v_dev, NY * NX * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phi[0][0],
                   phi_dev,
                   NY * NX * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&phi0[0][0],
                   phi0_dev,
                   NY * NX * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&rho[0][0],
                   rho_dev,
                   NY * NX * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(
            &p[0][0], p_dev, NY * NX * sizeof(double), cudaMemcpyDeviceToHost);

        datadeal(m);  // 计算宏观量，并输出宏观量
                      // X_s=A_spike( );
                      // X_b=A_bulble( );
        err1 = error1(phi, phi0);
        printf("t=%d err1=%e\n", m, err1);
      }
    }
    //	time_end=clock();
    //	printf("The time is: %f seconds\n",
    //(float)(time_end-time_begin)/CLOCKS_PER_SEC);
    cudaFree(f_dev);
    cudaFree(F_dev);
    cudaFree(g_dev);
    cudaFree(G_dev);
    cudaFree(phi_dev);
    cudaFree(rho_dev);
    cudaFree(p_dev);
    cudaFree(phi0_dev);
    cudaFree(mu_dev);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(u0_dev);
    cudaFree(v0_dev);
  }

  return 0;
}


void init()  //分布函数的初始化和flag的初始化
{
  double DDphi, mu0, uv, Ban, t_s;  // u_f流体f的平衡态速度 rho_f是流体f的密度
  double rhotal_1;                  //所有点的密度和
  int j, k, i, jp, kp;
  double Pe, x, h, lamda, wl, uu, niu, tau_f, tau_g;

  beta = 12.0 * sigma /
         (D * (phil - phig) * (phil - phig) * (phil - phig) * (phil - phig));
  Kappa = 1.5 * D * sigma / ((phil - phig) * (phil - phig));

  lamda = 256.0;
  wl = 256.0;
  uu = 0.04;
  ggy = -uu * uu / lamda;

  niu = uu * lamda / Re;
  tau_g = 3.0 * niu / (dx) + 0.5;
  tau_f = 0.8;
  ww_f = 1.0 / tau_f;
  ww_g = 1.0 / tau_g;

  Pe = 50.0;
  MM = uu * D / (beta * Pe * (phil - phig) * (phil - phig));
  A = 3.0 * MM / ((tau_f - 0.5) * dt);

  Ban = sigma * (2. * pi / lamda) * (2. * pi / lamda) / (-(rhol - rhog) * ggy);
  t_s = sqrt(-At * ggy / lamda);

  rhotal_1 = 0.;
  for (j = 1; j <= M + 1; j++) {
    for (k = N16; k <= N16 + N; k++) {
      // h=0.6*NY+0.05*wl*cos(2.0*pi*k/wl);
      h = 0.5 * NY + 0.05 * wl * cos(2.0 * pi * (k - N16) / wl);
      x = 2. * (j - h) / D;
      //	phi[j][k]=0.5*(phil+phig)+0.5*(phil-phig)*tanh(x);
      //	h=865+5.36*cos(2.0*pi*(k-N16)/lamda+3.0);
      // x=2.0*(j-h)/D;
      phi[j][k] = 0.5 * (phil + phig) + 0.5 * (phil - phig) * tanh(x);
      /*  if(j>=h&&j<=NY)
  {
    phi[j][k]=phil;
  }
  else
  {
    phi[j][k]=phig;
  }*/

      rho[j][k] = (phi[j][k] - phig) * (rhol - rhog) / (phil - phig) + rhog;

      p[j][k] = 0.0;
      u[j][k] = 0.0;
      v[j][k] = 0.0;

      uv = 1.5 * (u[j][k] * u[j][k] + v[j][k] * v[j][k]);

      u0[j][k] = u[j][k];
      v0[j][k] = v[j][k];
      phi0[j][k] = phi[j][k];


      DDphi = 0.0;
      for (i = 0; i < 9; i++) {
        jp = j + e[i][1];
        kp = (k + e[i][0] + N1 - N16) % N1 + N16;

        if (jp < 1 || jp > M1) {
          jp = j;
          kp = k;
        }

        DDphi += w[i] * phi[jp][kp];
      }
      mu0 = 4. * beta * (phi[j][k] - phil) * (phi[j][k] - phig) *
            (phi[j][k] - phim);
      mu[j][k] = mu0 - Kappa * (6.0 * rdt * rdt * (DDphi - phi[j][k]));

      f[0][j][k] = FEQ_0(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[1][j][k] = FEQ_1(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[3][j][k] = FEQ_2(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[2][j][k] = FEQ_3(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[4][j][k] = FEQ_4(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[5][j][k] = FEQ_5(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[6][j][k] = FEQ_6(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[7][j][k] = FEQ_7(phi[j][k], u[j][k], v[j][k], mu[j][k], A);
      f[8][j][k] = FEQ_8(phi[j][k], u[j][k], v[j][k], mu[j][k], A);

      g[0][j][k] = GEQ_0(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[1][j][k] = GEQ_1(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[3][j][k] = GEQ_2(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[2][j][k] = GEQ_3(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[4][j][k] = GEQ_4(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[5][j][k] = GEQ_5(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[6][j][k] = GEQ_6(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[7][j][k] = GEQ_7(rho[j][k], p[j][k], u[j][k], v[j][k], uv);
      g[8][j][k] = GEQ_8(rho[j][k], p[j][k], u[j][k], v[j][k], uv);

      rhotal_1 += phi[j][k];
    }
  }
  printf(
      "rhotal_1=%lf Re=%lf ggy=%e Pe=%lf tau_f=%lf tau_g=%lf niu=%e MM=%lf "
      "A=%lf D=%lf sigma=%lf beta=%lf Kappa=%lf Ban=%lf t_s=%e\n",
      rhotal_1,
      Re,
      ggy,
      Pe,
      tau_f,
      tau_g,
      niu,
      MM,
      A,
      D,
      sigma,
      beta,
      Kappa,
      Ban,
      t_s);
}

///////////////////////////////////////////

__global__ void collision_propagation(double A,
                                      double MM,
                                      double w_f,
                                      double w_g,
                                      double ggy,
                                      double *phi_d,
                                      double *mu_d,
                                      double *rho_d,
                                      double *p_d,
                                      double *u_d,
                                      double *v_d,
                                      double *phi0_d,
                                      double *u0_d,
                                      double *v0_d,
                                      double *f_d,
                                      double *F_d,
                                      double *g_d,
                                      double *G_d) {
  int x, y, k;

  double f0, f1, f2, f3, f4, f5, f6, f7, f8;  //当前点的分布函数
  double g0, g1, g2, g3, g4, g5, g6, g7, g8;
  double mf0, mf1, mf2, mf3, mf4, mf5, mf6, mf7, mf8;
  double mg0, mg1, mg2, mg3, mg4, mg5, mg6, mg7, mg8;
  double GG0, GG1, GG2, GG3, GG4, GG5, GG6, GG7, GG8;
  double mGG0, mGG1, mGG2, mGG3, mGG4, mGG5, mGG6, mGG7, mGG8;
  double s_f0, s_f1, s_f2, s_f3, s_f4, s_f5, s_f6, s_f7, s_f8;
  double s_g0, s_g1, s_g2, s_g3, s_g4, s_g5, s_g6, s_g7, s_g8;
  double wx0, wy0, wx1, wy1, wx2, wy2, wx3, wy3, wx4, wy4, wx5, wy5, wx6, wy6,
      wx7, wy7, wx8, wy8;
  double vv;
  double Drhox, Drhoy, Dphix, Dphiy, DDmu;
  double Fx, Fy;

  x = N16 + blockIdx.x * BX + threadIdx.x;  // f_d, F_d 矩阵中列下标
  y = 1 + blockIdx.y * BY;                  // f_d, F_d 矩阵中行下标
  k = NX * y + x;                           // f_d, F_d矩阵的一维offset

  if (x <= N + N16) {
    f0 = f_d[k + 0 * NYNX];
    f1 = f_d[k + 1 * NYNX];
    f2 = f_d[k + 2 * NYNX];
    f3 = f_d[k + 3 * NYNX];
    f4 = f_d[k + 4 * NYNX];
    f5 = f_d[k + 5 * NYNX];
    f6 = f_d[k + 6 * NYNX];
    f7 = f_d[k + 7 * NYNX];
    f8 = f_d[k + 8 * NYNX];

    g0 = g_d[k + 0 * NYNX];
    g1 = g_d[k + 1 * NYNX];
    g2 = g_d[k + 2 * NYNX];
    g3 = g_d[k + 3 * NYNX];
    g4 = g_d[k + 4 * NYNX];
    g5 = g_d[k + 5 * NYNX];
    g6 = g_d[k + 6 * NYNX];
    g7 = g_d[k + 7 * NYNX];
    g8 = g_d[k + 8 * NYNX];

    mf0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    mf1 = -4 * f0 - f1 - f2 - f3 - f4 + 2 * (f5 + f6 + f7 + f8);
    mf2 = 4 * f0 - 2 * (f1 + f2 + f3 + f4) + f5 + f6 + f7 + f8;
    mf3 = f1 - f3 + f5 - f6 - f7 + f8;
    mf4 = -2 * (f1 - f3) + f5 - f6 - f7 + f8;
    mf5 = f2 - f4 + f5 + f6 - f7 - f8;
    mf6 = -2 * (f2 - f4) + f5 + f6 - f7 - f8;
    mf7 = f1 - f2 + f3 - f4;
    mf8 = f5 - f6 + f7 - f8;

    mg0 = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;
    mg1 = -4 * g0 - g1 - g2 - g3 - g4 + 2 * (g5 + g6 + g7 + g8);
    mg2 = 4 * g0 - 2 * (g1 + g2 + g3 + g4) + g5 + g6 + g7 + g8;
    mg3 = g1 - g3 + g5 - g6 - g7 + g8;
    mg4 = -2 * (g1 - g3) + g5 - g6 - g7 + g8;
    mg5 = g2 - g4 + g5 + g6 - g7 - g8;
    mg6 = -2 * (g2 - g4) + g5 + g6 - g7 - g8;
    mg7 = g1 - g2 + g3 - g4;
    mg8 = g5 - g6 + g7 - g8;

    s_f3 = s_f5 = w_f;
    s_f0 = 1.0;
    s_f7 = s_f8 = 1.0;
    s_f1 = s_f2 = 1.3;
    s_f4 = s_f6 = 1.3;
    /*
 s_f3=s_f5=w_f;
   s_f0=w_f;  s_f7=s_f8=w_f;
 s_f1=s_f2=w_f;
 s_f4=s_f6=w_f;*/

    s_g0 = s_g3 = s_g5 = 1.0;
    s_g1 = 1.0;
    s_g2 = 1.0;
    s_g4 = s_g6 = 1.7;
    s_g7 = s_g8 = w_g;

    Drhox = Drhoy = 0.0;
    Dphix = Dphiy = 0.0;
    DDmu = 0.0;

    int xd = (x + 1 - N16 + N1) % N1 + N16;
    int xs = (x - 1 - N16 + N1) % N1 + N16;
    // 1方向
    Drhox += 1.0 / 9.0 * rho_d(y, xd);
    Dphix += 1.0 / 9.0 * phi_d(y, xd);
    DDmu += 1.0 / 9.0 * (mu_d(y, xd) - mu_d(y, x));

    // 3方向
    Drhox -= 1.0 / 9.0 * rho_d(y, xs);
    Dphix -= 1.0 / 9.0 * phi_d(y, xs);
    DDmu += 1.0 / 9.0 * (mu_d(y, xs) - mu_d(y, x));
    // 478方向
    if ((y - 1) >= 1) {
      Drhoy -= 1.0 / 9.0 * rho_d(y - 1, x);  // 4方向
      Dphiy -= 1.0 / 9.0 * phi_d(y - 1, x);
      DDmu += 1.0 / 9.0 * (mu_d(y - 1, x) - mu_d(y, x));

      Drhox -= 1.0 / 36.0 * rho_d(y - 1, xs);  // 7方向
      Drhoy -= 1.0 / 36.0 * rho_d(y - 1, xs);
      Dphix -= 1.0 / 36.0 * phi_d(y - 1, xs);
      Dphiy -= 1.0 / 36.0 * phi_d(y - 1, xs);

      DDmu += 1.0 / 36.0 * (mu_d(y - 1, xs) - mu_d(y, x));

      Drhox += 1.0 / 36.0 * rho_d(y - 1, xd);  // 8方向
      Drhoy -= 1.0 / 36.0 * rho_d(y - 1, xd);
      Dphix += 1.0 / 36.0 * phi_d(y - 1, xd);
      Dphiy -= 1.0 / 36.0 * phi_d(y - 1, xd);

      DDmu += 1.0 / 36.0 * (mu_d(y - 1, xd) - mu_d(y, x));
    } else {
      Drhoy -= 1.0 / 9.0 * rho_d(y, x);  // 4
      Dphiy -= 1.0 / 9.0 * phi_d(y, x);

      Drhox -= 1.0 / 36.0 * rho_d(y, x);  // 7
      Drhoy -= 1.0 / 36.0 * rho_d(y, x);
      Dphix -= 1.0 / 36.0 * phi_d(y, x);
      Dphiy -= 1.0 / 36.0 * phi_d(y, x);

      Drhox += 1.0 / 36.0 * rho_d(y, x);  // 8
      Drhoy -= 1.0 / 36.0 * rho_d(y, x);
      Dphix += 1.0 / 36.0 * phi_d(y, x);
      Dphiy -= 1.0 / 36.0 * phi_d(y, x);
    }
    // 256方向
    if ((y + 1) <= M1) {
      Drhoy += 1.0 / 9.0 * rho_d(y + 1, x);  // 2方向
      Dphiy += 1.0 / 9.0 * phi_d(y + 1, x);
      DDmu += 1.0 / 9.0 * (mu_d(y + 1, x) - mu_d(y, x));

      Drhox += 1.0 / 36.0 * rho_d(y + 1, xd);  // 5方向
      Drhoy += 1.0 / 36.0 * rho_d(y + 1, xd);
      Dphix += 1.0 / 36.0 * phi_d(y + 1, xd);
      Dphiy += 1.0 / 36.0 * phi_d(y + 1, xd);

      DDmu += 1.0 / 36.0 * (mu_d(y + 1, xd) - mu_d(y, x));

      Drhox -= 1.0 / 36.0 * rho_d(y + 1, xs);  // 6方向
      Drhoy += 1.0 / 36.0 * rho_d(y + 1, xs);
      Dphix -= 1.0 / 36.0 * phi_d(y + 1, xs);
      Dphiy += 1.0 / 36.0 * phi_d(y + 1, xs);

      DDmu += 1.0 / 36.0 * (mu_d(y + 1, xs) - mu_d(y, x));
    } else {
      Drhoy += 1.0 / 9.0 * rho_d(y, x);  // 2方向
      Dphiy += 1.0 / 9.0 * phi_d(y, x);

      Drhox += 1.0 / 36.0 * rho_d(y, x);  // 5方向
      Drhoy += 1.0 / 36.0 * rho_d(y, x);
      Dphix += 1.0 / 36.0 * phi_d(y, x);
      Dphiy += 1.0 / 36.0 * phi_d(y, x);

      Drhox -= 1.0 / 36.0 * rho_d(y, x);  // 6方向
      Drhoy += 1.0 / 36.0 * rho_d(y, x);
      Dphix -= 1.0 / 36.0 * phi_d(y, x);
      Dphiy += 1.0 / 36.0 * phi_d(y, x);
    }

    Fx = 3.0 * mu_d(y, x) * Dphix * rdt +
         3.0 * u_d(y, x) * (rhol - rhog) * MM * DDmu * rdt * rdt;  // Fx=Fsx+Fax
    Fy = 3.0 * mu_d(y, x) * Dphiy * rdt +
         3.0 * v_d(y, x) * (rhol - rhog) * MM * DDmu * rdt * rdt;

    vv = 1.5 * (u_d(y, x) * u_d(y, x) + v_d(y, x) * v_d(y, x));

    wx0 = (WEQ_0(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_0(u_d(y, x), v_d(y, x), vv) - 4.0 / 9.0) * rdt * Drhox);
    wy0 = (WEQ_0(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_0(u_d(y, x), v_d(y, x), vv) - 4.0 / 9.0) * rdt * Drhoy);
    wx1 = (WEQ_1(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_1(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhox);
    wy1 = (WEQ_1(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_1(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhoy);
    wx2 = (WEQ_2(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_2(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhox);
    wy2 = (WEQ_2(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_2(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhoy);
    wx3 = (WEQ_3(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_3(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhox);
    wy3 = (WEQ_3(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_3(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhoy);
    wx4 = (WEQ_4(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_4(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhox);
    wy4 = (WEQ_4(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_4(u_d(y, x), v_d(y, x), vv) - 1.0 / 9.0) * rdt * Drhoy);
    wx5 = (WEQ_5(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_5(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhox);
    wy5 = (WEQ_5(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_5(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhoy);
    wx6 = (WEQ_6(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_6(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhox);
    wy6 = (WEQ_6(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_6(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhoy);
    wx7 = (WEQ_7(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_7(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhox);
    wy7 = (WEQ_7(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_7(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhoy);
    wx8 = (WEQ_8(u_d(y, x), v_d(y, x), vv) * Fx +
           (WEQ_8(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhox);
    wy8 = (WEQ_8(u_d(y, x), v_d(y, x), vv) * (Fy + (rho_d(y, x) - rhom) * ggy) +
           (WEQ_8(u_d(y, x), v_d(y, x), vv) - 1.0 / 36.0) * rdt * Drhoy);

    GG0 = 3.0 * ((-u_d(y, x)) * wx0 + (-v_d(y, x)) * wy0);
    GG1 = 3.0 * ((1 - u_d(y, x)) * wx1 + (-v_d(y, x)) * wy1);
    GG2 = 3.0 * ((-u_d(y, x)) * wx2 + (1 - v_d(y, x)) * wy2);
    GG3 = 3.0 * ((-1 - u_d(y, x)) * wx3 + (-v_d(y, x)) * wy3);
    GG4 = 3.0 * ((-u_d(y, x)) * wx4 + (-1 - v_d(y, x)) * wy4);
    GG5 = 3.0 * ((1 - u_d(y, x)) * wx5 + (1 - v_d(y, x)) * wy5);
    GG6 = 3.0 * ((-1 - u_d(y, x)) * wx6 + (1 - v_d(y, x)) * wy6);
    GG7 = 3.0 * ((-1 - u_d(y, x)) * wx7 + (-1 - v_d(y, x)) * wy7);
    GG8 = 3.0 * ((1 - u_d(y, x)) * wx8 + (-1 - v_d(y, x)) * wy8);

    mGG0 = GG0 + GG1 + GG2 + GG3 + GG4 + GG5 + GG6 + GG7 + GG8;
    mGG1 = -4 * GG0 - GG1 - GG2 - GG3 - GG4 + 2 * (GG5 + GG6 + GG7 + GG8);
    mGG2 = 4 * GG0 - 2 * (GG1 + GG2 + GG3 + GG4) + GG5 + GG6 + GG7 + GG8;
    mGG3 = GG1 - GG3 + GG5 - GG6 - GG7 + GG8;
    mGG4 = -2 * (GG1 - GG3) + GG5 - GG6 - GG7 + GG8;
    mGG5 = GG2 - GG4 + GG5 + GG6 - GG7 - GG8;
    mGG6 = -2 * (GG2 - GG4) + GG5 + GG6 - GG7 - GG8;
    mGG7 = GG1 - GG2 + GG3 - GG4;
    mGG8 = GG5 - GG6 + GG7 - GG8;


    mf0 = diag0 *
          (mf0 -
           s_f0 * (mf0 -
                   MFEQ_0(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f0) * MFF_0(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf1 = diag1 *
          (mf1 -
           s_f1 * (mf1 -
                   MFEQ_1(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f1) * MFF_1(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf2 = diag2 *
          (mf2 -
           s_f2 * (mf2 -
                   MFEQ_2(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f2) * MFF_2(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf3 = diag3 *
          (mf3 -
           s_f3 * (mf3 -
                   MFEQ_3(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f3) * MFF_3(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf4 = diag4 *
          (mf4 -
           s_f4 * (mf4 -
                   MFEQ_4(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f4) * MFF_4(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf5 = diag5 *
          (mf5 -
           s_f5 * (mf5 -
                   MFEQ_5(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f5) * MFF_5(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf6 = diag6 *
          (mf6 -
           s_f6 * (mf6 -
                   MFEQ_6(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f6) * MFF_6(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf7 = diag7 *
          (mf7 -
           s_f7 * (mf7 -
                   MFEQ_7(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f7) * MFF_7(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));
    mf8 = diag8 *
          (mf8 -
           s_f8 * (mf8 -
                   MFEQ_8(phi_d(y, x), u_d(y, x), v_d(y, x), mu_d(y, x), A)) +
           (1 - 0.5 * s_f8) * MFF_8(phi_d(y, x),
                                    u_d(y, x),
                                    v_d(y, x),
                                    phi0_d(y, x),
                                    u0_d(y, x),
                                    v0_d(y, x)));

    mg0 = diag0 *
          (mg0 -
           s_g0 * (mg0 - MGEQ_0(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g0) * mGG0);
    mg1 = diag1 *
          (mg1 -
           s_g1 * (mg1 - MGEQ_1(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g1) * mGG1);
    mg2 = diag2 *
          (mg2 -
           s_g2 * (mg2 - MGEQ_2(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g2) * mGG2);
    mg3 = diag3 *
          (mg3 -
           s_g3 * (mg3 - MGEQ_3(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g3) * mGG3);
    mg4 = diag4 *
          (mg4 -
           s_g4 * (mg4 - MGEQ_4(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g4) * mGG4);
    mg5 = diag5 *
          (mg5 -
           s_g5 * (mg5 - MGEQ_5(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g5) * mGG5);
    mg6 = diag6 *
          (mg6 -
           s_g6 * (mg6 - MGEQ_6(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g6) * mGG6);
    mg7 = diag7 *
          (mg7 -
           s_g7 * (mg7 - MGEQ_7(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g7) * mGG7);
    mg8 = diag8 *
          (mg8 -
           s_g8 * (mg8 - MGEQ_8(rho_d(y, x), p_d(y, x), u_d(y, x), v_d(y, x))) +
           dt * (1 - 0.5 * s_g8) * mGG8);

    f0 = mf0 - 4 * (mf1 - mf2);
    f1 = mf0 - mf1 - 2 * (mf2 + mf4) + mf3 + mf7;
    f2 = mf0 - mf1 - 2 * (mf2 + mf6) + mf5 - mf7;
    f3 = mf0 - mf1 - 2 * (mf2 - mf4) - mf3 + mf7;
    f4 = mf0 - mf1 - 2 * (mf2 - mf6) - mf5 - mf7;
    f5 = mf0 + mf1 + mf1 + mf2 + mf3 + mf4 + mf5 + mf6 + mf8;
    f6 = mf0 + mf1 + mf1 + mf2 - mf3 - mf4 + mf5 + mf6 - mf8;
    f7 = mf0 + mf1 + mf1 + mf2 - mf3 - mf4 - mf5 - mf6 + mf8;
    f8 = mf0 + mf1 + mf1 + mf2 + mf3 + mf4 - mf5 - mf6 - mf8;

    g0 = mg0 - 4 * (mg1 - mg2);
    g1 = mg0 - mg1 - 2 * (mg2 + mg4) + mg3 + mg7;
    g2 = mg0 - mg1 - 2 * (mg2 + mg6) + mg5 - mg7;
    g3 = mg0 - mg1 - 2 * (mg2 - mg4) - mg3 + mg7;
    g4 = mg0 - mg1 - 2 * (mg2 - mg6) - mg5 - mg7;
    g5 = mg0 + mg1 + mg1 + mg2 + mg3 + mg4 + mg5 + mg6 + mg8;
    g6 = mg0 + mg1 + mg1 + mg2 - mg3 - mg4 + mg5 + mg6 - mg8;
    g7 = mg0 + mg1 + mg1 + mg2 - mg3 - mg4 - mg5 - mg6 + mg8;
    g8 = mg0 + mg1 + mg1 + mg2 + mg3 + mg4 - mg5 - mg6 - mg8;


    // 0 1 3方向
    F_d[k] = f0;
    G_d[k] = g0;
    F_d(1, y, xd) = f1;
    G_d(1, y, xd) = g1;
    F_d(3, y, xs) = f3;
    G_d(3, y, xs) = g3;
    // 2 5 6方向
    if ((y + 1) <= M1) {
      F_d(2, y + 1, x) = f2;
      G_d(2, y + 1, x) = g2;
      F_d(5, y + 1, xd) = f5;
      G_d(5, y + 1, xd) = g5;
      F_d(6, y + 1, xs) = f6;
      G_d(6, y + 1, xs) = g6;
    } else {
      F_d(4, y, x) = f2;
      G_d(4, y, x) = g2;
      F_d(7, y, x) = f5;
      G_d(7, y, x) = g5;
      F_d(8, y, x) = f6;
      G_d(8, y, x) = g6;
    }
    // 4 7 8方向
    if ((y - 1) >= 1) {
      F_d(4, y - 1, x) = f4;
      G_d(4, y - 1, x) = g4;
      F_d(7, y - 1, xs) = f7;
      G_d(7, y - 1, xs) = g7;
      F_d(8, y - 1, xd) = f8;
      G_d(8, y - 1, xd) = g8;
    } else {
      F_d(2, y, x) = f4;
      G_d(2, y, x) = g4;
      F_d(5, y, x) = f7;
      G_d(5, y, x) = g7;
      F_d(6, y, x) = f8;
      G_d(6, y, x) = g8;
    }
  }
}

__global__ void Macro_rho(double *phi_d,
                          double *phi0_d,
                          double *rho_d,
                          double *f_d) {
  int x, y, k;
  double f0, f1, f2, f3, f4, f5, f6, f7, f8;

  x = N16 + blockIdx.x * BX + threadIdx.x;  // f_d, F_d 矩阵中列下标x
  y = 1 + blockIdx.y * BY;                  // f_d, F_d 矩阵中行下标y
  k = NX * y + x;

  if (x <= N + N16) {
    f0 = f_d[k + 0 * NYNX];
    f1 = f_d[k + 1 * NYNX];
    f2 = f_d[k + 2 * NYNX];
    f3 = f_d[k + 3 * NYNX];
    f4 = f_d[k + 4 * NYNX];
    f5 = f_d[k + 5 * NYNX];
    f6 = f_d[k + 6 * NYNX];
    f7 = f_d[k + 7 * NYNX];
    f8 = f_d[k + 8 * NYNX];

    phi0_d(y, x) = phi_d(y, x);

    phi_d(y, x) = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    rho_d(y, x) = 0.5 * (phi_d(y, x) - phig) * (rhol - rhog) + rhog;
  }
}

__global__ void Macro_mu(double Kappa,
                         double beta,
                         double *phi_d,
                         double *mu_d) {
  int x, y;

  double DDphi, mu0;

  x = N16 + blockIdx.x * BX + threadIdx.x;  // f_d, F_d 矩阵中列下标x
  y = 1 + blockIdx.y * BY;                  // f_d, F_d 矩阵中行下标y


  if (x <= N + N16) {
    int xd = (x + 1 - N16 + N1) % N1 + N16;
    int xs = (x - 1 - N16 + N1) % N1 + N16;

    DDphi = 0.0;
    DDphi += 1.0 / 9.0 * (phi_d(y, xd) - phi_d(y, x));  // 1方向
    DDphi += 1.0 / 9.0 * (phi_d(y, xs) - phi_d(y, x));  // 3方向
    if ((y + 1) <= M1) {
      DDphi += 1.0 / 9.0 * (phi_d(y + 1, x) - phi_d(y, x));    // 2方向
      DDphi += 1.0 / 36.0 * (phi_d(y + 1, xd) - phi_d(y, x));  // 5方向
      DDphi += 1.0 / 36.0 * (phi_d(y + 1, xs) - phi_d(y, x));  // 6方向
    }
    if ((y - 1) >= 1) {
      DDphi += 1.0 / 9.0 * (phi_d(y - 1, x) - phi_d(y, x));    // 4方向
      DDphi += 1.0 / 36.0 * (phi_d(y - 1, xs) - phi_d(y, x));  // 7方向
      DDphi += 1.0 / 36.0 * (phi_d(y - 1, xd) - phi_d(y, x));  // 8方向
    }

    mu0 = 4 * beta * (phi_d(y, x) - phil) * (phi_d(y, x) - phig) *
          (phi_d(y, x) - phim);
    mu_d(y, x) = mu0 - 6.0 * Kappa * rdt * rdt * DDphi;
  }
}

__global__ void Macro_u(double ggy,
                        double MM,
                        double *phi_d,
                        double *rho_d,
                        double *p_d,
                        double *mu_d,
                        double *u_d,
                        double *v_d,
                        double *u0_d,
                        double *v0_d,
                        double *g_d) {
  int x, y, k;

  double g1, g2, g3, g4, g5, g6, g7, g8;

  double Drhox, Drhoy, Dphix, Dphiy, DDmu;
  double UX, UY, rF, FFa, s0u, udrho, Fsx, Fsy;

  x = N16 + blockIdx.x * BX + threadIdx.x;  // f_d, F_d 矩阵中列下标x
  y = 1 + blockIdx.y * BY;                  // f_d, F_d 矩阵中行下标y
  k = NX * y + x;

  if (x <= N + N16) {
    //	g0 = g_d[k+0*NYNX];
    g1 = g_d[k + 1 * NYNX];
    g2 = g_d[k + 2 * NYNX];
    g3 = g_d[k + 3 * NYNX];
    g4 = g_d[k + 4 * NYNX];
    g5 = g_d[k + 5 * NYNX];
    g6 = g_d[k + 6 * NYNX];
    g7 = g_d[k + 7 * NYNX];
    g8 = g_d[k + 8 * NYNX];

    u0_d(y, x) = u_d(y, x);
    v0_d(y, x) = v_d(y, x);

    int xd = (x + 1 - N16 + N1) % N1 + N16;
    int xs = (x - 1 - N16 + N1) % N1 + N16;

    Drhox = Drhoy = 0.0;
    Dphix = Dphiy = DDmu = 0.0;

    Drhox += 1.0 / 9.0 * rho_d(y, xd);  // 1方向
    Dphix += 1.0 / 9.0 * phi_d(y, xd);
    DDmu += 1.0 / 9.0 * (mu_d(y, xd) - mu_d(y, x));

    // 3方向
    Drhox -= 1.0 / 9.0 * rho_d(y, xs);
    Dphix -= 1.0 / 9.0 * phi_d(y, xs);
    DDmu += 1.0 / 9.0 * (mu_d(y, xs) - mu_d(y, x));
    // 478方向
    if ((y - 1) >= 1) {
      Drhoy -= 1.0 / 9.0 * rho_d(y - 1, x);  // 4方向
      Dphiy -= 1.0 / 9.0 * phi_d(y - 1, x);
      DDmu += 1.0 / 9.0 * (mu_d(y - 1, x) - mu_d(y, x));

      Drhox -= 1.0 / 36.0 * rho_d(y - 1, xs);  // 7方向
      Drhoy -= 1.0 / 36.0 * rho_d(y - 1, xs);
      Dphix -= 1.0 / 36.0 * phi_d(y - 1, xs);
      Dphiy -= 1.0 / 36.0 * phi_d(y - 1, xs);

      DDmu += 1.0 / 36.0 * (mu_d(y - 1, xs) - mu_d(y, x));


      Drhox += 1.0 / 36.0 * rho_d(y - 1, xd);  // 8方向
      Drhoy -= 1.0 / 36.0 * rho_d(y - 1, xd);
      Dphix += 1.0 / 36.0 * phi_d(y - 1, xd);
      Dphiy -= 1.0 / 36.0 * phi_d(y - 1, xd);

      DDmu += 1.0 / 36.0 * (mu_d(y - 1, xd) - mu_d(y, x));
    } else {
      Drhoy -= 1.0 / 9.0 * rho_d(y, x);  // 4
      Dphiy -= 1.0 / 9.0 * phi_d(y, x);

      Drhox -= 1.0 / 36.0 * rho_d(y, x);  // 7
      Drhoy -= 1.0 / 36.0 * rho_d(y, x);
      Dphix -= 1.0 / 36.0 * phi_d(y, x);
      Dphiy -= 1.0 / 36.0 * phi_d(y, x);

      Drhox += 1.0 / 36.0 * rho_d(y, x);  // 8
      Drhoy -= 1.0 / 36.0 * rho_d(y, x);
      Dphix += 1.0 / 36.0 * phi_d(y, x);
      Dphiy -= 1.0 / 36.0 * phi_d(y, x);
    }
    // 256方向
    if ((y + 1) <= M1) {
      Drhoy += 1.0 / 9.0 * rho_d(y + 1, x);  // 2方向
      Dphiy += 1.0 / 9.0 * phi_d(y + 1, x);
      DDmu += 1.0 / 9.0 * (mu_d(y + 1, x) - mu_d(y, x));

      Drhox += 1.0 / 36.0 * rho_d(y + 1, xd);  // 5方向
      Drhoy += 1.0 / 36.0 * rho_d(y + 1, xd);
      Dphix += 1.0 / 36.0 * phi_d(y + 1, xd);
      Dphiy += 1.0 / 36.0 * phi_d(y + 1, xd);

      DDmu += 1.0 / 36.0 * (mu_d(y + 1, xd) - mu_d(y, x));

      Drhox -= 1.0 / 36.0 * rho_d(y + 1, xs);  // 6方向
      Drhoy += 1.0 / 36.0 * rho_d(y + 1, xs);
      Dphix -= 1.0 / 36.0 * phi_d(y + 1, xs);
      Dphiy += 1.0 / 36.0 * phi_d(y + 1, xs);

      DDmu += 1.0 / 36.0 * (mu_d(y + 1, xs) - mu_d(y, x));
    } else {
      Drhoy += 1.0 / 9.0 * rho_d(y, x);  // 2方向

      Dphiy += 1.0 / 9.0 * phi_d(y, x);
      Drhox += 1.0 / 36.0 * rho_d(y, x);  // 5方向
      Drhoy += 1.0 / 36.0 * rho_d(y, x);
      Dphix += 1.0 / 36.0 * phi_d(y, x);
      Dphiy += 1.0 / 36.0 * phi_d(y, x);

      Drhox -= 1.0 / 36.0 * rho_d(y, x);  // 6方向
      Drhoy += 1.0 / 36.0 * rho_d(y, x);
      Dphix -= 1.0 / 36.0 * phi_d(y, x);
      Dphiy += 1.0 / 36.0 * phi_d(y, x);
    }

    p_d(y, x) = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;

    UX = g1 - g3 + g5 + g8 - g6 - g7;
    UY = g2 - g4 + g5 + g6 - g7 - g8;

    Fsx = 3.0 * mu_d(y, x) * rdt * Dphix;
    Fsy = 3.0 * mu_d(y, x) * rdt * Dphiy;

    FFa = 1.5 * (rhol - rhog) * MM * (rdt * DDmu);

    rF = 1.0 / (rho_d(y, x) - FFa);
    u_d(y, x) = (UX + 0.5 * dt * Fsx) * rF;
    v_d(y, x) = (UY + 0.5 * dt * (Fsy + (rho_d(y, x) - rhom) * ggy)) * rF;

    s0u = -2.0 / 3.0 * (u_d(y, x) * u_d(y, x) + v_d(y, x) * v_d(y, x));
    udrho = 3.0 * (u_d(y, x) * Drhox + v_d(y, x) * Drhoy);
    p_d(y, x) = 0.6 * (p_d(y, x) + 0.5 * udrho + rho_d(y, x) * s0u);
  }
}

double error1(double phi[NY][NX], double phi0[NY][NX]) {
  int x, y;
  double temp1 = 0, temp2 = 0, error;

  for (y = 1; y <= M1; y++) {
    for (x = N16; x <= N16 + N; x++) {
      temp1 += fabs(phi[y][x] - phi0[y][x]);
      temp2 += fabs(phi[y][x]);
    }
  }
  error = temp1 / temp2;
  return (error);
}

void datadeal(int t)  //计算、输出宏观量
{
  int x, y;
  int Reint;
  double rhotal_2;
  FILE *fp;
  char filename[50];

  Reint = int(Re);
  sprintf(filename, "./Output/%s%.4d%s%.8d", "Re", Reint, "phi", t);

  rhotal_2 = 0;
  for (y = 1; y <= M1; y++) {
    for (x = N16; x <= N16 + N; x++) {
      rhotal_2 += phi[y][x];
    }
  }
  printf("rhotal_2=%lf\n", rhotal_2);

  fp = fopen(filename, "w");
  for (y = 1; y <= M1; y++) {
    for (x = N16; x <= N16 + N; x++) {
      fprintf(fp, "%e ", phi[y][x]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  /*
sprintf(filename,"%s%d","ux",t);
  fp=fopen(filename,"w");
  for(y=1;y<=M+1;y++)
{
for(x=N16;x<=N16+N;x++)
{
  fprintf(fp,"%e ",ux[y][x]);
}
 fprintf(fp,"\n");
}
fclose(fp);

sprintf(filename,"%s%d","uy",t);
  fp=fopen(filename,"w");
  for(y=1;y<=M+1;y++)
{
for(x=N16;x<=N16+N;x++)
{
  fprintf(fp,"%e ",uy[y][x]);
}
fprintf(fp,"\n");
}
fclose(fp);*/
}
/*
 double A_spike( )
 {
   int j, k, flag;
     double ls;

   for(k=N16;k<=N16+N;k++)
   {
     for(j=1;j<=M1;j++)
     {
       if(phi[j][k]>=-0.01&&phi[j][k]<=0.01)
       {
         ls=NY/2.0-j;
         flag=1;
         break;
       }

     }
     if(flag==1)
     {
       break;
     }
   }

   return ls;
 }

double A_bulble( )
{
  int j, k, flag;
  double lb;
  for(k=N16;k<=N16+N;k++)
  {
         for(j=M1;j>=1;j--)
     {
       if(phi[j][k]>=-0.01&&phi[j][k]<=0.01)
       {
         lb=j-NY/2.0;
         flag=1;
         break;
       }
     }
     if(flag==1)
     {
       break;
     }
   }
   return lb;
 }*/
