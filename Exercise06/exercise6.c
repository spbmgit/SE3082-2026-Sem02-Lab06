#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP 8   // strip size aligned to typical SIMD width (8 doubles with AVX-512)

int main() {
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0;
        B[i] = (i % 100) * 1.0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i += STRIP) {
        int end = (i + STRIP < N) ? i + STRIP : N;
        #pragma omp simd
        for (int j = i; j < end; j++) {
            C[j] = A[j] * B[j];
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("Time taken: %f seconds\n", elapsed);
    printf("C[0] = %f, C[N-1] = %f\n", C[0], C[N-1]);

    free(A);
    free(B);
    free(C);
    return 0;
}
