#include <x86intrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>  

#define SHELLS 101
#define PHOTONS 163840

float heat[SHELLS] = {0.0f};
float heat2[SHELLS] = {0.0f};

static void photon(int threads)
{
    #pragma omp parallel
    {
        int cph = 0;
        float p_heat[SHELLS] = {0.0f};
        float p_heat2[SHELLS] = {0.0f};

        while (cph < PHOTONS / threads)
        {
            p_heat[0]++;
            p_heat2[0]++;

            //#pragma omp atomic
            cph++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < SHELLS; i++) {
                heat[i] += p_heat[i];
                heat2[i] += p_heat2[i];
            }
        }
    }
}

int main()
{
    int threads = omp_get_max_threads();
    printf("hilos = %d\n", threads);

    photon(threads);
    printf("heat[0] = %f\n", heat[0]);
    printf("heat2[0] = %f\n", heat2[0]);

    return 0;
}
