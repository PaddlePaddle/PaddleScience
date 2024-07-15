#include "common.h"
#include "lb.h"

// save the results
void datasave() {
  int x, y;
  double ut, vt, rhot;
  char u_name[20], v_name[20];
  FILE *fp1, *fp2, *fp3;

  sprintf(u_name, "%s%.8d", "./Output/u", n);
  sprintf(v_name, "%s%.8d", "./Output/v", n);
  if ((fp1 = fopen(u_name, "w")) == NULL) return;
  if ((fp2 = fopen(v_name, "w")) == NULL) return;
  if ((fp3 = fopen("./Output/rho", "w")) == NULL) return;
  /*
if((fp1=fopen("./Output/u","w")) == NULL) return;
if((fp2=fopen("./Output/v","w")) == NULL) return;
if((fp3=fopen("./Output/rho","w")) == NULL) return;
  */

  for (y = 1; y <= M1; y++) {
    for (x = N16; x < N16 + N1; x++) {
      ut = UX(y, x) + 0.5 * dt * Fx;
      vt = VY(y, x);
      rhot = RHO(y, x);

      fprintf(fp1, "%e ", ut);
      fprintf(fp2, "%e ", vt);
      fprintf(fp3, "%e ", rhot);
    }
    fprintf(fp1, "\n\n");
    fprintf(fp2, "\n\n");
    fprintf(fp3, "\n\n");
  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
