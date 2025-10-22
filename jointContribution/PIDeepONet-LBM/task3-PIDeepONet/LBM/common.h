#ifndef __COMMON_H_
#define __COMMON_H_

#define XL 1
#define M (256 * XL)     // Grid in y- direciton
#define N (256 * XL)     // Grid in x- direciton
#define M1 (M + 1)       // Number of the grid in y- direction
#define N1 (N + 1)       // Number of the grid in x- direction
#define Ly 1.f           // Length of the domain in y- direction
#define Lx (Ly * N / M)  // Length of the domain in x- direction
//------------------------------------------------------------------------------------------GPU
#define N16 16
#define NX2 ((N1 / 16 + 1) * 16 + N16)
#define NY2 (M + 3)
#define NYNX2 (NY2 * NX2)
#define Mc (NY2 / 2)
#define Nc (NX2 / 2)
#define Mb 1
#define Me M1
////////////////////////////////////////////////////////////////////////////////////////////////////
#define BX 128
#define BY 1
#define NT 64  // used for boundary grid
#define BCX 64
#define BCY 1
////////////////////////////////////////////////////////////////////////////////////////////////////
#define Q 9
#define T 1000
#define PI (4.0 * atan(1.0))

// parameters used in physical field
extern double nu, U0;
extern double rho0;
extern double rho_in, rho_out;
extern double Fx, Fy;
// dimensionless parameters
extern double Re;
// parameters used in LBE simulation
extern double tau_f, wf, ci, rcc;
// parameters used in computation
extern int n, nmax;
extern double dx, dt, dn;
extern double sum_u_o;

/////////////////////////////////////////////////////////////////////////////////////////////////
// CPU
extern double f[Q][NY2][NX2];
extern double sf[Q];
//---------------------------------------------------------------------------------------------
// device adress
extern double *f_dev;
extern double *F_dev;
extern double *sf_dev;
//-----------------------------------------------------------------------------------------------------------------
#endif
