#include <cuda.h>
#include <cstdio>
#include <cstdint>

#include "helper_cuda.h"
#include "wtime.h"
#include "params.h"

template <typename T>
T div_ceil(T a, T b) {
    return (a + b - 1) / b;
}


__global__ void photon(float * heat, uint32_t * s) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < 4; i++) {
        s[i] = idx + 1;
    }

    int thread_from = WORK * (blockDim.x * blockIdx.x + threadIdx.x);
    int thread_to = min(WORK * (blockDim.x * blockIdx.x + threadIdx.x + 1), PHOTONS);

    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;

    float _heat[SHELLS] = {0};

    for (int idx = thread_from; idx < thread_to; ++idx)
    {
        for (;;) {
            float t = -logf(rand01()); /* move */
            x += t * u;
            y += t * v;
            z += t * w;

            unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
            if (shell > SHELLS - 1) {
                shell = SHELLS - 1;
            }

            _heat[shell] += (1.0f - albedo) * weight;
            weight *= albedo;

            /* New direction, rejection method */
            float xi1, xi2;
            do {
                xi1 = 2.0f * rand01() - 1.0f;
                xi2 = 2.0f * rand01() - 1.0f;
                t = xi1 * xi1 + xi2 * xi2;
            } while (1.0f < t); 
            u = 2.0f * t - 1.0f;
            v = xi1 * sqrtf((1.0f - u * u) / t);
            w = xi2 * sqrtf((1.0f - u * u) / t);

            if (weight < 0.001f) { /* roulette */
                if (rand01() > 0.1f)
                    break;
                weight /= 0.1f;
            }
        }
    }

    for (int i = 0; i < SHELLS; i++) {
        atomicAdd(&heat[i], _heat[i]);
    }
}

int main() {
    uint32_t s[4];

    float heat[SHELL] = {0};
    float heat2[SHELL] = {0};

    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(PHOTONS, BLOCK_SIZE * WORK));

    photon<<<grid, block>>>(heat, s);

    for (int i = 0; i < SHELLS; i++) {
        heat2[i] = heat[i] * heat[i];
    }

    return 0;
}