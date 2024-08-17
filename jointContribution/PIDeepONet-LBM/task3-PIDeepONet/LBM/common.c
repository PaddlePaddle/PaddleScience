#include "common.h"
#include <stdio.h>
#include <stdlib.h>

// parameters used in physical field
double nu, U0;
double rho0 = 1.0;
double rho_in, rho_out;
double Fx, Fy;
// dimensionless parameters
double Re = 1000.0;
// parameters used in LBE simulation
double tau_f, wf, ci, rcc;
// parameters used in computation
int n = 0, nmax;
double dx, dt, dn;
double sum_u_o = 0.0;

// CPU
double f[Q][NY2][NX2];
double g[Q][NY2][NX2];
double sf[Q];
// device adress
double *f_dev;
double *F_dev;
double *sf_dev;
