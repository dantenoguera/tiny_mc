#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

extern void seed();
extern __m256 rand01();
extern __m256 fastlogf_simd(__m256 x);

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];

void load_heat(unsigned int* shell, float* ht, float* ht2)
{
    for (int i = 0; i < 8; i++)
    {
        heat[shell[i]] += ht[i];
        heat2[shell[i]] += ht2[i];
    }
}

int count_photons(unsigned int* rmask)
{
    int count = 0;
    for(int i = 0; i < 8; i++)
    {
        if (rmask[i] == 0xFFFFFFFF)
            count++;
    }

    return count;
}

/***
 * Photon
 ***/

static void photon(void)
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
    __m256 x = _mm256_set1_ps(0.0);
    __m256 y = _mm256_set1_ps(0.0);
    __m256 z = _mm256_set1_ps(0.0);
    __m256 u = _mm256_set1_ps(0.0);
    __m256 v = _mm256_set1_ps(0.0);
    __m256 w = _mm256_set1_ps(1.0);

    __m256 weight = _mm256_set1_ps(1.0);

    int i = 0;
    while (i < PHOTONS) {
        /* move */
        __m256 t = -logf(rand01()); // TODO

        x = _mm256_fmadd_ps(t, u, x); // x = t * u + x
        y = _mm256_fmadd_ps(t, v, y);
        z = _mm256_fmadd_ps(t, w, z);
        
        /* absorb */
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 z2 = _mm256_mul_ps(z, z);

        __m256 x2py2 = _mm256_add_ps(x2, y2);
        __m256 tot = _mm256_add_ps(x2py2, z2);
        __m256 rad = _mm256_sqrt_ps(tot);

        __m256i shell = _mm256_cvttps_epu32(_mm256_mul_ps(rad, shells_per_mfp));
        __m256i max_shell = _mm256_set1_epi32(SHELLS - 1);
        
        shell = _mm256_min_epi32(shell, max_shell);

        // shell = [ph1, ph2, ph3, ..., ph8]
        //ph1 = 1
        //ph2 = 1
        // heat[1] += ...
        // heat[1] += ...
        // ...
        // heat[]

        __m256 ht = _mm256_fnmadd_ps(weight, albedo, weight); // (1.0f - albedo) * weight
        __m256 ht2 = _mm256_mul_ps(ht,ht); // (1.0f - albedo) * (1.0f - albedo) * weight * weight
        load_heat((unsigned int *)&shell, (float*)&ht, (float*)&ht2);
        weight *= albedo; // usar intrinseca?
        
        /* roulette */
        __m256 wmask = _mm256_cmp_ps(weight, zzo, _CMP_LT_OS);
        __m256 weightX10 = _mm256_mul_ps(weight, ten);
        weight = _m256_blendv_ps(weight, weightX10, wmask);

        __m256 rmask = _mm256_cmp_ps(rand01(), zo, _CMP_GT_OS);
        weight = _m256_blendv_ps(weight, one, rmask);
        x = _m256_blendv_ps(x, zero, rmask);
        y = _m256_blendv_ps(y, zero, rmask);
        z = _m256_blendv_ps(z, zero, rmask);

        i += count_photons((unsigned int*)&rmask);


        /* New direction, rejection method */
        float xi1, xi2; // TODO
        do {
            xi1 = 2.0f * rand01() - 1.0f;
            xi2 = 2.0f * rand01() - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);

        u = _mm256_fmsub_ps(two, t, one); // u = 2*t-1
        
        __m256 sqr = _mm256_sqrt_ps(_mm256_div_ps(_mm256_fnmadd_ps(u, u, one), t)); // sqrtf((1.0f - u * u)
        
        v = _mm256_mul_ps(xi1, sqr);
        w = _mm256_mul_ps(xi2, sqr);
    }
}

/***
 * Main matter
 ***/

int main(void)
{
    // configure RNG
    seed(SEED);
    // start timer
    double start = wtime();
    // simulation
    photon();

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
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
#else
    printf("%lf", 1e-3 * PHOTONS / elapsed);
#endif

    return 0;
}
