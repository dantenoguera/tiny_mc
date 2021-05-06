#include <time.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <x86intrin.h>

#define SEED (time(NULL))

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return 1e-9 * ts.tv_nsec + (double)ts.tv_sec;
}

static __m256i s[4];

static inline __m256i rotl(const __m256i x, int k) {
    __m256i xklshift = _mm256_slli_epi32(x, k);
    __m256i x32krshift = _mm256_srli_epi32(x, 32 - k);

    return _mm256_or_si256(xklshift, x32krshift);
}


static __m256i next(void) {
    const __m256i result = _mm256_add_epi32(s[0], s[3]);
    const __m256i t = _mm256_slli_epi32(s[1], 9);

    s[2] = _mm256_xor_si256(s[2], s[0]);
    s[3] = _mm256_xor_si256(s[3], s[1]);
    s[1] = _mm256_xor_si256(s[1], s[2]);
    s[0] = _mm256_xor_si256(s[0], s[3]);
    s[2] = _mm256_xor_si256(s[2], t);

    s[3] = rotl(s[3], 11);

    return result;
}

static __m256i mix32(__m256i x) {
    __m256i a = _mm256_set1_epi32(0x9e3779b9);
    __m256i b = _mm256_set1_epi32(0xbf58476d);
    __m256i c = _mm256_set1_epi32(0x94d049bb);

    x = _mm256_add_epi32(x, a);

    __m256i z = x;

    z = _mm256_mul_epi32(_mm256_xor_si256(z,
        _mm256_srli_epi32(z, 14)), b);

    z = _mm256_mul_epi32(_mm256_xor_si256(z,
        _mm256_srli_epi32(z, 11)), c);

    return _mm256_xor_si256(z,
        _mm256_srli_epi32(z, 15));
}

void seed(__m256i sd) {
    s[0] = mix32(sd);
    s[1] = mix32(s[0]);
    s[2] = mix32(s[1]);
    s[3] = mix32(s[2]);
}

__m256 rand01() {
    __m256i _n = _mm256_abs_epi32(next());
    __m256 uint_max = _mm256_set1_ps((float)UINT_MAX);
    __m256 n = _mm256_cvtepi32_ps(_n) / uint_max;

    return n;
}

int main() {

    srand(SEED);

    int _sd[8];
    for (int i = 0; i < 8; i++)
        _sd[i] = rand();

    __m256i sd = _mm256_set_epi32(_sd[0] , _sd[1], _sd[2], _sd[3],
        _sd[4], _sd[5], _sd[6], _sd[7]);

    seed(sd);

    __m256 a;
    double start = wtime();
    for (int i = 0; i < 1000000000 / 8; i++)
    {
        a = rand01();
    }
    double end = wtime();
    assert(start <= end);

    printf("last value: %f %f %f %f %f %f %f %f \n", 
        a[0], a[1] , a[2], a[3], a[4], a[5], a[6], a[7]);
    printf("elapsed: %lf\n", end - start);
    return 0;
}
