#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NPOINTS 2000
#define MAXITER 2000

struct complex{
  double real;
  double imag;
};

int main(int argc, char *argv[]){
  int numoutside = 0;
  double area, error;

  int nthreads = 4;
  if (argc > 1) nthreads = atoi(argv[1]);
  omp_set_num_threads(nthreads);

  double start = omp_get_wtime();

  #pragma omp parallel reduction(+:numoutside)
  {
    int i, j, iter;
    double ztemp;
    struct complex z, c;

    int tid = omp_get_thread_num();
    int nthr = omp_get_num_threads();

    for (i = tid; i < NPOINTS; i += nthr) {
      for (j = 0; j < NPOINTS; j++) {
        c.real = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + 1.0e-7;
        c.imag = 1.125 * (double)(j) / (double)(NPOINTS) + 1.0e-7;
        z = c;
        for (iter = 0; iter < MAXITER; iter++) {
          ztemp = (z.real*z.real) - (z.imag*z.imag) + c.real;
          z.imag = z.real*z.imag*2 + c.imag;
          z.real = ztemp;
          if ((z.real*z.real + z.imag*z.imag) > 4.0e0) {
            numoutside++;
            break;
          }
        }
      }
    }
  }

  double elapsed = omp_get_wtime() - start;

  area = 2.0*2.5*1.125*(double)(NPOINTS*NPOINTS - numoutside) / (double)(NPOINTS*NPOINTS);
  error = area / (double)NPOINTS;

  printf("Threads = %d\n", nthreads);
  printf("Area of Mandelbrot set = %12.8f +/- %12.8f\n", area, error);
  printf("Time taken = %f seconds\n", elapsed);

  return 0;
}
