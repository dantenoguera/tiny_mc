#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

/***
 * xoshiro128+
 * https://prng.di.unimi.it/xoshiro128plus.c
 ***/

static uint32_t s[4];

static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t next(void) {
    const uint32_t result = s[0] + s[3];

    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    return result;
}

static float float_from_uint32(uint32_t i) {
    return (i >> 8) * 0x1.0p-24f;
}

static uint32_t mix32(uint32_t x) {
    uint32_t z = (x += 0x9e3779b9);
    z = (z ^ (z >> 14)) * 0xbf58476d;
    z = (z ^ (z >> 11)) * 0x94d049bb;
    return z ^ (z >> 15);
}

static void seed(uint32_t sd) {
    s[0] = mix32(sd);
    s[1] = mix32(s[0]);
    s[2] = mix32(s[1]);
    s[3] = mix32(s[2]);
}

static float rand01() {
    return float_from_uint32(next());
}

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static __m256 heat[SHELLS];
static __m256 heat2[SHELLS];


/***
 * Photon
 ***/

static void photon(void)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float _shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    __m256 x;
    __m256 y;
    __m256 z;
    __m256 u;
    __m256 v;
    __m256 w;

    x = _mm256_xor_ps(x, x);
    y = _mm256_xor_ps(y, z);
    z = _mm256_xor_ps(z, z);
    u = _mm256_xor_ps(u, u);
    v = _mm256_xor_ps(v, v);

    float _w = 1.0f;
    w = _mm256_broadcast_ss(&_w);
    
    float weight = 1.0f;

    for (;;) {
        /* move */
        float _t = -logf(rand01()); 
        __m256 t = _mm256_broadcast_ss(&_t); // mal

        x = _mm256_fmadd_ps(t, u, x); // x = t * u + x
        y = _mm256_fmadd_ps(t, v, y);
        z = _mm256_fmadd_ps(t, w, z);
        
        /* absorb */
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 z2 = _mm256_mul_ps(z, z);

        __m256 x2my2 = _mm256_add_ps(x2, y2);
        __m256 tot = _mm256_add_ps(x2my2, z2);

        __m256 rad = _mm256_sqrt_ps(tot);
        __m256 shells_per_mfp = _mm256_broadcast_ss(&_shells_per_mfp);

        __m256i shell = _mm256_cvttps_epu32(_mm256_mul_ps(rad, shells_per_mfp));
        if (shell > SHELLS - 1) { // TODO
            shell = SHELLS - 1;
        }

        // TODO
        // shell = [ph1, ph2, ph3, ..., ph8]
        //ph1 = 1
        //ph2 = 1
        // heat[1] += ...
        // heat[1] += ...
        // ...
        // heat[]

        heat[shell] += (1.0f - albedo) * weight;
        heat2[shell] += (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
        weight *= albedo;

        /* roulette */
        if (weight < 0.001f) { // TODO
            if (rand01() > 0.1f)
                break;
            weight /= 0.1f;
        }

        /* New direction, rejection method */
        float xi1, xi2; // TODO
        do {
            xi1 = 2.0f * rand01() - 1.0f;
            xi2 = 2.0f * rand01() - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        
        float _one = 1.0f;
        __m256 one = _mm256_broadcast_ss(1.0f);
        __m256 v = _mm256_set1_ps(1.0f);

        float _two = 2.0f;
        __m256 two = _mm256_broadcast_ss(&_two);

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
    for (unsigned int i = 0; i < PHOTONS; ++i) { // TODO: incrementar de a 8 o tenerlo en cuenta
        photon();
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
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
#else
    printf("%lf", 1e-3 * PHOTONS / elapsed);
#endif

    return 0;
}
