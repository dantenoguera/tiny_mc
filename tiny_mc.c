#include "params.h"
#include "wtime.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>
#include <stdbool.h>
#include <omp.h>

extern void seed(__m256i sd);
extern __m256 rand01();
extern __m256 fastlogf_simd(__m256 x);

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";

// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];
#pragma omp threadprivate(heat, heat2)

static inline void load_heat(unsigned int * const restrict shell, 
        float * const restrict ht, float * const restrict ht2) {
    for (int i = 0; i < 8; i++) {
        heat[shell[i]] += ht[i];
        heat2[shell[i]] += ht2[i];
    }
}

static inline int count_set_bits(int n) {
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }

    return count;
}

/*
 TODO: 
    - heats para c/thread
    - estado s para c/thread
    - cph atomic?
*/
/***
 * Photon
 ***/
static void photon(int threads)
{
    const __m256 zero = _mm256_set1_ps(0.0);
    const __m256 zzo = _mm256_set1_ps(0.001);
    const __m256 zo = _mm256_set1_ps(0.1);
    const __m256 one = _mm256_set1_ps(1.0);
    const __m256 two = _mm256_set1_ps(2.0);
    const __m256 ten = _mm256_set1_ps(10.0);
    const __m256 albedo = _mm256_set1_ps(MU_S / (MU_S + MU_A));
    const __m256 shells_per_mfp = _mm256_set1_ps(1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));

    /* launch */
    __m256 x = zero;
    __m256 y = zero;
    __m256 z = zero;
    __m256 u = zero;
    __m256 v = zero;
    __m256 w = one;
    __m256 weight = one;

    int cph = 0; // count simulated photons
    int nphotons = _nphotons[omp_get_thread_num()];
    while (cph < nphotons) {
        /* move */
        __m256 t = -fastlogf_simd(rand01());
        x = _mm256_fmadd_ps(t, u, x); // x = t * u + x
        y = _mm256_fmadd_ps(t, v, y); 
        z = _mm256_fmadd_ps(t, w, z); 
        
        /* absorb */
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 z2 = _mm256_mul_ps(z, z);
        __m256 sum = _mm256_add_ps(x2, _mm256_add_ps(y2, z2));
        __m256 rad = _mm256_sqrt_ps(sum);

        __m256i volatile shell = _mm256_cvttps_epi32(
            _mm256_mul_ps(rad, shells_per_mfp));

        __m256i max_shell = _mm256_set1_epi32(SHELLS - 1);
        shell = _mm256_min_epi32(shell, max_shell); // shell = min(shell, SHELLS - 1)
        
        __m256 ht = _mm256_fnmadd_ps(weight, albedo, weight); // (1.0f - albedo) * weight
        __m256 ht2 = _mm256_mul_ps(ht, ht); // (1.0f - albedo) * (1.0f - albedo) * weight * weight
        load_heat((unsigned int *)&shell, (float*)&ht, (float*)&ht2); // sacar?

        weight = _mm256_mul_ps(weight, albedo);

        /* roulette */
        __m256 wmask = _mm256_cmp_ps(weight, zzo, _CMP_LT_OS);

        __m256 weightX10 = _mm256_mul_ps(weight, ten);
        weight = _mm256_blendv_ps(weight, weightX10, wmask);

        __m256 rmask = _mm256_cmp_ps(rand01(), zo, _CMP_GT_OS); 

        __m256 breakmask = _mm256_and_ps(wmask, rmask);

        weight = _mm256_blendv_ps(weight, one, breakmask);

        x = _mm256_blendv_ps(x, zero, breakmask);
        y = _mm256_blendv_ps(y, zero, breakmask);
        z = _mm256_blendv_ps(z, zero, breakmask);

        cph += count_set_bits(_mm256_movemask_ps(breakmask));

        /* New direction, rejection method */
        __m256 xi1 = zero;
        __m256 xi2 = zero; 
        __m256 bigtmask = _mm256_cmp_ps(one, one, _CMP_TRUE_US); // prende todo el vector
        do {
            __m256 _xi1 = _mm256_fmsub_ps(two, rand01(), one); // 2.0f * rand - 1.0f
            __m256 _xi2 = _mm256_fmsub_ps(two, rand01(), one);

            xi1 = _mm256_blendv_ps(xi1, _xi1, bigtmask);
            xi2 = _mm256_blendv_ps(xi2, _xi2, bigtmask);

            __m256 _t = _mm256_add_ps(_mm256_mul_ps(xi1, xi1),
                _mm256_mul_ps(xi2, xi2));

            t = _mm256_blendv_ps(t, _t, bigtmask);

            bigtmask = _mm256_cmp_ps(one, t, _CMP_LT_OS); // T si 1.0 < t

        } while (_mm256_movemask_ps(bigtmask));

        u = _mm256_fmsub_ps(two, t, one); // u = 2 * t - 1
        
        __m256 srt = _mm256_sqrt_ps(
            _mm256_div_ps(_mm256_fnmadd_ps(u, u, one), t)); // sqrtf((1.0f - u * u)
        
        v = _mm256_mul_ps(xi1, srt);
        w = _mm256_mul_ps(xi2, srt);
    
    }
}

/***
 * Main matter
 ***/
int main(void)
{
    // configure xoshiro128+
    const int threads = omp_get_max_threads();
#if VERBOSE
    printf("threads = %d\n", threads);
#endif
    
    omp_sched_t kind;
    int chunk_size;
    omp_get_schedule(&kind, &chunk_size);

#if VERBOSE
    printf("schedule: ");
    switch(kind)
    {
        case omp_sched_static:
            printf("static");
            break;
        case omp_sched_dynamic:
            printf("dynamic");
            break;
        case omp_sched_guided:
            printf("guided");
            break;
        case omp_sched_auto:
            printf("auto");
            break;
        default:
            printf("other (implementation specific)");
            break;
    }
    printf("\nchunks: %d\n", chunk_size);
#endif
 
    __m256i seeds[threads];

    srand(SEED);
    for (int i = 0; i < threads; i++) {
        seeds[i] = _mm256_set_epi32(
                rand(), rand(), rand(), rand(),
                rand(), rand(), rand(), rand());
    }
    
    int _res = PHOTONS / threads;
    int _rem = PHOTONS % threads;
    int _nphotons[threads];
    for( unsigned int k = 0; k < threads; k++ ){
      _nphotons[k] = _res;
    }
    for( unsigned int i = 0; i < _rem; i++ ){
      _nphotons[i] += 1;
    }
    for( unsigned int j=0; j < threads; j++ ){
      printf("%d\n", _nphotons[j]);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        seed(seeds[tid]);
    }

    // start timer
    double start = wtime();
    // simulation

    float heat_ac[SHELLS] = {0.0f};
    float heat2_ac[SHELLS] = {0.0f};

    #pragma omp parallel reduction(+ : heat_ac, heat2_ac)
    {
        photon(_nphotons);

        for (int i = 0; i < SHELLS; i++) {
            heat_ac[i] += heat[i];
            heat2_ac[i] += heat2[i];
        }
    }

    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

#if VERBOSE
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);
    printf("# %lf seconds\n", elapsed);
    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");

    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat_ac[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2_ac[i] - heat_ac[i] * heat_ac[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat_ac[SHELLS - 1] / PHOTONS);
#else
    printf("%lf", 1e-3 * PHOTONS / elapsed);
#endif

    return 0;
}
