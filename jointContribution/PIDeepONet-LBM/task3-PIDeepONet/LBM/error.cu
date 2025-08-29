#include "common.h"
#include "lb.h"

double error() {
  int x, y;
  double sum_u_c = 0.0;
  double Err;

  for (y = 1; y <= M1; y++)
    for (x = N16; x < N16 + N1; x++) {
      sum_u_c += UX(y, x);
    }

  Err = fabs(sum_u_c - sum_u_o) / fabs(sum_u_c);
  sum_u_o = sum_u_c;

  return Err;
}
