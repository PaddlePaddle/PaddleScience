//---LBE for 2D cavity
// flow----------------------------------------------------------
//---@Author: Xuhui
// Meng-------------------------------------------------------------
#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"
#include "datasave.cu"
#include "error.cu"
#include "flow.cu"
#include "init.cu"
#include "lb.h"
//----------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  clock_t time_begin, time_end;
  double uc, vc;
  int new_step, goon;
  double err = 1.0;

  int device = 1;
  cudaSetDevice(device);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  printf("Lattice Boltzmann Simulation running on: %s\n", properties.name);

  dim3 threads(BX, 1);
  dim3 grid((N1 + BX - 1) / BX, M1);
  dim3 gridBlr((M1 + NT - 1) / NT, 2);
  dim3 gridBub((N1 + NT - 1) / NT, 2);

  // parameters
  init();
  datasave();
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // GPU memory: f_dev[], F_dev[]
  cudaMalloc((void **)&f_dev, sizeof(double) * Q * NY2 * NX2);
  cudaMalloc((void **)&F_dev, sizeof(double) * Q * NY2 * NX2);
  cudaMalloc((void **)&sf_dev, sizeof(double) * Q);

  // from CPU to GPU (GPU <= CPU): f_dev <= f
  cudaMemcpy(f_dev,
             &f[0][0][0],
             sizeof(double) * Q * NY2 * NX2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(sf_dev, &sf, sizeof(double) * Q, cudaMemcpyHostToDevice);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

loop:
  printf("Enter the num of steps:");
  scanf("%d", &new_step);
  nmax += new_step;
  printf("nmax = %d\n", nmax);

  dn = 9. / nmax;

  time_begin = clock();
  while (n < nmax)  //&&(err > 1.0e-5))
  {
    // Excute kernel collision_propagation : f => F F => f
    Evol_flow<<<grid, threads>>>(sf_dev, dt, Fx, Fy, f_dev, F_dev);
    Bc_flow_X<<<gridBlr, NT>>>(dt, Fx, Fy, U0, F_dev);
    Bc_flow_Y<<<gridBub, NT>>>(dt, Fx, Fy, U0, n, dn, F_dev);

    n += 1;

    Evol_flow<<<grid, threads>>>(sf_dev, dt, Fx, Fy, F_dev, f_dev);
    Bc_flow_X<<<gridBlr, NT>>>(dt, Fx, Fy, U0, f_dev);
    Bc_flow_Y<<<gridBub, NT>>>(dt, Fx, Fy, U0, n, dn, f_dev);

    n += 1;

    if (n % T == 0) {
      cudaMemcpy(&f[0][0][0],
                 f_dev,
                 Q * NY2 * NX2 * sizeof(double),
                 cudaMemcpyDeviceToHost);
      uc = UX(Mc, Nc);
      vc = VY(Mc, Nc);
      err = error();
      printf("n=%d: err = %.3e, uc = %.3e, vc = %.3e\n", n, err, uc, vc);
      datasave();
    }
  }

  time_end = clock();
  printf("\nThe computing time is: %f seconds ",
         (double)(time_end - time_begin) / CLOCKS_PER_SEC);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (n % 2 == 0) {
    cudaMemcpy(&f[0][0][0],
               f_dev,
               Q * NY2 * NX2 * sizeof(double),
               cudaMemcpyDeviceToHost);
    printf("this is from f !\n");
  } else {
    cudaMemcpy(&f[0][0][0],
               F_dev,
               Q * NY2 * NX2 * sizeof(double),
               cudaMemcpyDeviceToHost);
    printf("this is from F !\n");
  }

  // save data
  datasave();

  printf("goon? yes(1) no(0):");
  scanf("%d", &goon);
  if (goon) goto loop;


  // free GPU memory
  cudaFree(f_dev);
  cudaFree(F_dev);
  cudaFree(sf_dev);
  ////////////////////////////////////////////////////////////////////////////////////////////////_GPU
  return 0;
}
