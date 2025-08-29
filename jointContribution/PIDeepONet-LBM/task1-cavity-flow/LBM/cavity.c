// modified Ladd's method: direct force
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define M 200
#define N 200
#define M2 (M / 2)
#define N2 (N / 2)
#define M1 (M + 1)
#define N1 (N + 1)
#define Ly (1.0)

#define tau0 (0.8)
#define A0 (0.1)
#define dp (0.0)

#define n_PL 1.0
//#define mu_PL 0.0018
#define Re 100.0
#define rho0 1.0
#define U0 0.1

#define Q 9

void lbini(void);
void analy(void);
void data_read(void);
void Evol(void);
void geo(void);
double feq(int k, int y, int x);
double force(int k, int y, int x);
void datadeal(void);
void Mass_velo_error(void);
double f[M1][N1][Q], g[M1][N1][Q], u[M1][N1], v[M1][N1], p[M1][N1], A[M1][N1],
    tau[M1][N1], Sh[M1][N1][3], uans[M1];
double umax, utem0, utem1, utem2, u0, mu_PL;
int m, e[Q][2], re[Q];
int flag[M1][N1];
double tp[Q], s[Q], diag[Q];
double c, rc, rcc, dx, dt, cs2;
double w, w1, rho_in, rho_out;
double Err_mass, Err_vel;


void main() {
  int readdata, mmax, TEND = 0;
  double drho, err;

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
  tp[0] = 4.0 / 9;
  tp[1] = tp[2] = tp[3] = tp[4] = 1.0 / 9;
  tp[5] = tp[6] = tp[7] = tp[8] = 1.0 / 36;

  re[0] = 0;
  re[1] = 3;
  re[2] = 4;
  re[3] = 1;
  re[4] = 2;
  re[5] = 7;
  re[6] = 8;
  re[7] = 5;
  re[8] = 6;

  c = 1.0;
  dx = Ly / (M);
  dt = dx / c;
  rc = 1.0 / c;
  rcc = rc * rc;
  cs2 = c * c / 3;

  // drho=N*dx*dp/cs2;   rho_in=rho0+0.5*drho;  rho_out=rho0-0.5*drho;

  mu_PL = rho0 * pow(U0, 2. - n_PL) / Re;
  w = 1.0 / tau0;
  w1 = 1.0 - 0.5 * w;
  // A0=tau-0.5-mu_PL/rho0/(cs2*dt);

  utem0 = 1.0 / mu_PL * dp;
  utem1 = n_PL / (n_PL + 1) * pow(utem0, 1.0 / n_PL);
  utem2 = pow(0.5 * Ly, 1.0 + 1.0 / n_PL);
  umax = utem1 * utem2;

  diag[0] = 1.0 / 9;
  diag[1] = diag[2] = 1.0 / 36;
  diag[3] = diag[5] = 1.0 / 6;
  diag[4] = diag[6] = 1.0 / 12;
  diag[7] = diag[8] = 1.0 / 4;

  printf(
      "mu_ori=%e, A0=%e, pindex=%f, dx=%e, dt=%e\n", mu_PL, A0, n_PL, dx, dt);

  // geo();
  lbini();
  analy();

  // printf("Read Data? (yes=1 no=0)\n");
  // scanf("%d",&readdata);
  // if(readdata) data_read();

  m = 0;
  err = 1.0;

AA:
  printf("input mmax:\n");
  scanf("%d", &mmax);
  //  mmax=1000000;
  TEND += mmax;

  u0 = u[M2][N2];
  while (m < TEND && err > 1.0e-9) {
    m++;
    Evol();

    if (m % 500 == 0) {
      err = fabs(u[M2][N2] - u0) / (fabs(u[M2][N2]) + 1.0e-10);
      u0 = u[M2][N2];
      printf("err=%e ucenter=%e  m=%d\n", err, u[M2][N2], m);
    }

    if (m % 2000 == 0) {
      Mass_velo_error();
      printf("err_mass=%e, err_velo=%e\n", Err_mass, Err_vel);
      datadeal();
    }
  }

  Mass_velo_error();
  printf("err_mass=%e, err_velo=%e\n", Err_mass, Err_vel);
  datadeal();


  printf("Continue? (yes=1 no=0)\n");
  scanf("%d", &readdata);
  if (readdata) goto AA;
}

void lbini() {
  int i, j, k;

  for (j = 0; j < M1; j++)
    for (i = 0; i < N1; i++) {
      u[j][i] = 0.0;
      v[j][i] = 0.0;
      p[j][i] = rho0;
      A[j][i] = A0;
      tau[j][i] = tau0;
      Sh[j][i][0] = Sh[j][i][1] = Sh[j][i][2] = 0.0;
    }

  for (j = 0; j <= M; j++)
    for (i = 0; i <= N; i++)
      for (k = 0; k < Q; k++) f[j][i][k] = feq(k, j, i);
}

void geo() {
  int i, j;
  for (j = 1; j < M; j++) {
    for (i = 1; i < N; i++) {
      flag[j][i] = 1;
    }
  }
}

void analy() {
  int j;
  double yd, yabs;
  FILE *fp;

  for (j = 0; j < M1; j++) {
    yd = (j - 0.5) * dx - 0.5 * Ly;
    yabs = fabs(yd);
    uans[j] = umax - utem1 * pow(yabs, 1.0 + 1.0 / n_PL);
  }
  //--save data---
  if ((fp = fopen("uans.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }

  for (j = 0; j < M1; j++) {
    fprintf(fp, "%e", uans[j] / umax);
    fprintf(fp, "\n");
  }
  fclose(fp);

  printf("datasave completed!\n");
}

void Mass_velo_error() {
  int i, j, nt = 0;
  double p_sum = 0.0, p_nu_sum = 0.0, u_ana_sum = 0.0, u_ana_nu_sum = 0.0;

  for (j = 0; j < M1; j++)
    for (i = 0; i < N1; i++) {
      p_sum += rho0;
      p_nu_sum += fabs(p[j][i] - rho0);
    }

  Err_mass = p_nu_sum / p_sum;

  for (j = 0; j <= M; j++)
    for (i = 0; i < N1; i++) {
      u_ana_sum += fabs(uans[j]);
      u_ana_nu_sum += fabs(u[j][i] - uans[j]);
    }

  Err_vel = u_ana_nu_sum / u_ana_sum;
}

/*
double feq(int k, int y, int x)
{
  double RHO, U, V, At;
  double uv, eu, f1eq, eqf;
  double sheq,seq00,seq01,seq02,seq11,f2eq;

  RHO=rho[y][x]; U=u[y][x]; V=v[y][x]; At=A[y][x];
  eu=(e[k][0]*U+e[k][1]*V)*rc;
  uv=(U*U+V*V)*rcc;
  f1eq= 1.0+3*eu+4.5*eu*eu-1.5*uv;

  seq00=Sh[y][x][0]*(e[k][0]*e[k][0]-1.0/3);
  seq01=Sh[y][x][1]* e[k][0]*e[k][1] ;
  seq11=Sh[y][x][2]*(e[k][1]*e[k][1]-1.0/3);
  sheq=seq00+2.0*seq01+seq11;
  f2eq=1.5*At*dt*sheq;

  eqf=tp[k]*RHO*(f1eq+f2eq);

  return eqf;
}
*/

double feq(int k, int y, int x) {
  double PHO, U, V, At;
  double uv, eu, f1eq, eqf;
  double sheq, seq00, seq01, seq02, seq11, f2eq;

  PHO = p[y][x];
  U = u[y][x];
  V = v[y][x];
  At = A[y][x];
  eu = (e[k][0] * U + e[k][1] * V) * rc;
  uv = (U * U + V * V) * rcc;
  f1eq = 3 * eu + 4.5 * eu * eu - 1.5 * uv;

  // eqf = tp[k]*(PHO + rho0*f1eq);

  seq00 = Sh[y][x][0] * (e[k][0] * e[k][0] - 1.0 / 3);
  seq01 = Sh[y][x][1] * e[k][0] * e[k][1];
  seq11 = Sh[y][x][2] * (e[k][1] * e[k][1] - 1.0 / 3);
  sheq = seq00 + 2.0 * seq01 + seq11;
  f2eq = 1.5 * At * dt * sheq;

  eqf = tp[k] * PHO + rho0 * tp[k] * (f1eq + f2eq);
  /*
  if(k==0) eqf=rho0-(1-tp[k])*PHO/cs2+rho0*tp[k]*(f1eq+f2eq);
  else     eqf=tp[k]*PHO/cs2+rho0*tp[k]*(f1eq+f2eq);
  */

  return eqf;
}

double force(int k, int y, int x) {
  double F1, F2, Fc, U, V, rtau;

  U = u[y][x];
  V = v[y][x];
  rtau = 1.0 / tau[y][x];

  F1 = 3.0 * dp * e[k][0];
  F2 = 9.0 * (U * e[k][0] + V * e[k][1]) * (dp * e[k][0]) - 3.0 * U * dp;
  Fc = tp[k] * w1 * (F1 * rc + F2 * rcc);

  return Fc;
}

void Evol() {
  int i, j, k, id, jd;
  double FM, FCM, r = 10., wc, fneq;
  double cst, miut, at00, at01, at10, at11, att;
  double sum_rho_in_0 = 0.0, sum_rho_out_0 = 0.0;
  double alpha_in, alpha_out, FMu, FMb;

  // relaxation
  for (j = 0; j < M1; j++)
    for (i = 0; i < N1; i++) {
      for (k = 0; k < Q; k++) {
        FM = feq(k, j, i);
        FCM = force(k, j, i);
        g[j][i][k] = f[j][i][k] - w * (f[j][i][k] - FM) + dt * FCM;
      }
    }

  /*
  // Halfway bounce-back
  j=0;
  for(i=0;i<=N;i++)
   {
     for(k=0;k<Q;k++)
   {
      jd=j-e[k][1]; id=(i-e[k][0]+N1)%N1;
      if(jd>=0) f[j][i][k]=g[jd][id][k];
        else f[j][i][k]=g[j][i][re[k]];
   }

   }

  j=M;
  for(i=0;i<=N;i++)
  {
    for(k=0;k<Q;k++)
   {
    jd=j-e[k][1];  id=(i-e[k][0]+N1)%N1;
        if(jd<=M) f[j][i][k]=g[jd][id][k];
        else  f[j][i][k]=g[j][i][re[k]];
   }
  }
  */

  // stream
  for (j = 1; j < M; j++)
    for (i = 1; i < N; i++) {
      for (k = 0; k < Q; k++) {
        jd = (j - e[k][1]);
        id = (i - e[k][0]);
        f[j][i][k] = g[jd][id][k];
      }
    }

  // macroscopic
  for (j = 1; j < M; j++)
    for (i = 1; i < N; i++) {
      p[j][i] = f[j][i][0] + f[j][i][1] + f[j][i][2] + f[j][i][3] + f[j][i][4] +
                f[j][i][5] + f[j][i][6] + f[j][i][7] + f[j][i][8];
      u[j][i] = (f[j][i][1] + f[j][i][5] + f[j][i][8] - f[j][i][3] -
                 f[j][i][6] - f[j][i][7]);
      v[j][i] = (f[j][i][5] + f[j][i][6] + f[j][i][2] - f[j][i][7] -
                 f[j][i][8] - f[j][i][4]);

      u[j][i] = u[j][i] * c / rho0 + 0.5 * dt / rho0 * dp;
      v[j][i] = v[j][i] * c / rho0;
      /*
      F0 = -w1*tp[0]*(u[j][i]*dp)/cs2;
      Eps = Eps + tau*dt*F0 -
      rho0*tp[0]/cs2*0.5*(u[j][i]*u[j][i]+v[j][i]*v[j][i]);
      p[j][i]=cs2/(1. - tp[0])*Eps;
      */
    }

  //---left and right
  for (j = 0; j < M1; j++) {
    u[j][0] = 0.0;
    v[j][0] = 0.0;
    p[j][0] = p[j][1];
    A[j][0] = A[j][1];
    Sh[j][0][0] = Sh[j][1][0];
    Sh[j][0][1] = Sh[j][1][1];
    Sh[j][0][2] = Sh[j][1][2];

    u[j][N] = 0.0;
    v[j][N] = 0.0;
    p[j][N] = p[j][N - 1];
    A[j][N] = A[j][N - 1];
    Sh[j][N][0] = Sh[j][N - 1][0];
    Sh[j][N][1] = Sh[j][N - 1][1];
    Sh[j][N][2] = Sh[j][N - 1][2];

    for (k = 0; k < Q; k++) {
      FMu = feq(k, j, 0);
      f[j][0][k] = FMu + f[j][1][k] - feq(k, j, 1);

      FMb = feq(k, j, N);
      f[j][N][k] = FMb + f[j][N - 1][k] - feq(k, j, N - 1);
    }
  }

  //---upper and bottom
  for (i = 0; i < N1; i++) {
    j = 0;
    u[j][i] = 0.0;
    v[j][i] = 0.0;
    p[j][i] = p[j + 1][i];
    A[j][i] = A[j + 1][i];
    Sh[j][i][0] = Sh[j + 1][i][0];
    Sh[j][i][1] = Sh[j + 1][i][1];
    Sh[j][i][2] = Sh[j + 1][i][2];

    j = M;
    if (m <= 1000)
      u[j][i] = U0;
    else
      u[j][i] = U0 * (1. - cosh(r * (i * dx - 0.5)) / cosh(0.5 * r));
    v[j][i] = 0.0;
    p[j][i] = p[j - 1][i];
    A[j][i] = A[j - 1][i];
    Sh[j][i][0] = Sh[j - 1][i][0];
    Sh[j][i][1] = Sh[j - 1][i][1];
    Sh[j][i][2] = Sh[j - 1][i][2];

    for (k = 0; k < Q; k++) {
      j = 0;
      FMu = feq(k, j, i);
      f[j][i][k] = FMu + f[j + 1][i][k] - feq(k, j + 1, i);

      j = M;
      FMb = feq(k, j, i);
      f[j][i][k] = FMb + f[j - 1][i][k] - feq(k, j - 1, i);
    }
  }


  // compute the shear rate
  for (j = 0; j < M1; j++)
    for (i = 0; i < N1; i++) {
      at00 = at01 = at11 = 0.0;
      cst = rho0 * cs2 * (A[j][i] - tau0) * dt;

      for (k = 0; k < Q; k++) {
        at00 += e[k][0] * e[k][0] * f[j][i][k];
        at01 += e[k][0] * e[k][1] * f[j][i][k];
        at11 += e[k][1] * e[k][1] * f[j][i][k];
      }


      at00 *= c * c;
      at01 *= c * c;
      at11 *= c * c;

      at00 -= rho0 * u[j][i] * u[j][i] + p[j][i] * cs2;
      at01 -= rho0 * u[j][i] * v[j][i];
      at11 -= rho0 * v[j][i] * v[j][i] + p[j][i] * cs2;

      at00 += dt * (u[j][i] * dp);
      at01 += 0.5 * dt * (v[j][i] * dp);
      // at11+=0/rho[j][i];

      at00 /= cst;
      at01 /= cst;
      at10 = at01;
      at11 /= cst;

      Sh[j][i][0] = at00;
      Sh[j][i][1] = at01;
      Sh[j][i][2] = at11;

      att = at00 * at00 + at01 * at01 + at10 * at10 + at11 * at11;
      att /= 2;
      att = sqrt(att);

      miut = mu_PL * pow(att, n_PL - 1.0);
      // tau[j][i]=0.5+miut/(cs2*rho0*dt);
      A[j][i] = tau0 - 0.5 - miut / (cs2 * rho0 * dt);
    }
  /*
  //---inlet and outlet---density modified
  for(j=0; j<=M; j++)
    {
        rho[j][0]=2.0*rho[j][1]-rho[j][2];
      rho[j][N]=2.0*rho[j][N-1]-rho[j][N-2];

      sum_rho_in_0+=rho[j][0];
      sum_rho_out_0+=rho[j][N];

   }

    alpha_in = (M+1)*rho_in/sum_rho_in_0;
    alpha_out= (M+1)*rho_out/sum_rho_out_0;

   for (j=0; j<=M; j++)
    {
       rho[j][0]*=alpha_in;
     rho[j][N]*=alpha_out;
    }
  */
}

void datadeal() {
  int i, j;

  FILE *fp;

  if ((fp = fopen("uh.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  for (j = 0; j < M1; j++) {
    for (i = 0; i <= N; i++) fprintf(fp, "%e ", u[j][i]);
    fprintf(fp, "\n");
  }
  fclose(fp);

  if ((fp = fopen("vh.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  for (j = 0; j < M1; j++) {
    for (i = 0; i <= N; i++) fprintf(fp, "%e ", v[j][i]);
    fprintf(fp, "\n");
  }
  fclose(fp);

  if ((fp = fopen("p.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  for (j = 0; j <= M; j++) {
    for (i = 0; i <= N; i++) fprintf(fp, "%e ", p[j][i]);
    fprintf(fp, "\n");
  }
  fclose(fp);

  if ((fp = fopen("tau.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  for (j = 0; j <= M; j++) {
    for (i = 0; i <= N; i++) fprintf(fp, "%e ", A[j][i]);
    fprintf(fp, "\n");
  }
  fclose(fp);

  if ((fp = fopen("err.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  fprintf(fp, "%e  %e", Err_mass, Err_vel);
  // fprintf(fp,"\n");

  fclose(fp);

  if ((fp = fopen("flag.dat", "w")) == NULL) {
    printf(" File Open Error\n");
    exit(1);
  }
  for (j = 0; j <= M; j++) {
    for (i = 0; i <= N; i++) fprintf(fp, "%d ", flag[j][i]);
    fprintf(fp, "\n");
  }
  fclose(fp);


  fp = fopen("ut", "wb");
  fwrite(u, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("vt", "wb");
  fwrite(v, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("pt", "wb");
  fwrite(p, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("ft", "wb");
  fwrite(f, sizeof(double), M1 * N1 * Q, fp);
  fclose(fp);
}

void data_read() {
  FILE *fp;
  fp = fopen("ut", "rb");
  fread(u, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("vt", "rb");
  fread(v, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("pt", "rb");
  fread(p, sizeof(double), M1 * N1, fp);
  fclose(fp);
  fp = fopen("ft", "rb");
  fread(f, sizeof(double), M1 * N1 * Q, fp);
  fclose(fp);
}
