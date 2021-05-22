#include <x86intrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>  

#define SHELLS 101
#define PHOTONS 32768

float heat[SHELLS] = {0.0f};
float heat2[SHELLS] = {0.0f};
#pragma omp threadprivate(heat, heat2)

static void photon(int threads)
{
    int cph = 0;

    while (cph < PHOTONS / threads)
    {
        heat[0]++;
        heat2[0]++;

        cph++;
    }
}

int main()
{
    int threads = omp_get_max_threads();
    printf("hilos = %d\n", threads);

    float ht[SHELLS] = {0.0f};
    float ht2[SHELLS] = {0.0f};

    #pragma omp parallel reduction(+ : ht, ht2)
    {
        photon(threads);

        for (int i = 0; i < SHELLS; i++) {
            ht[i] += heat[i];
            ht2[i] += heat2[i];
        }
    }

    // photon(threads);
    printf("heat[0] = %f\n", heat[0]);
    printf("heat2[0] = %f\n", heat2[0]);
    printf("ht[0] = %f\n", ht[0]);
    printf("ht2[0] = %f\n", ht2[0]);

    return 0;
}
