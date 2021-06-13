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

//__device__ uint32_t s[4];

__global__ void photon(float * heat) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // seed PRNG
    // for (int i = 0; i < 4; i++) {
    //     s[i] = idx + 1;
    // }

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

    float _heat[SHELLS] = {0}; // local heat

    for (int idx = thread_from; idx < thread_to; ++idx)
    {
        for (;;) {
            /* move */
            float t = -logf(rand01());
            x += t * u;
            y += t * v;
            z += t * w;

            /* absorb */
            unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp;
            if (shell > SHELLS - 1) {
                shell = SHELLS - 1;
            }

            _heat[shell] += (1.0f - albedo) * weight;
            weight *= albedo;

            /* roulette */
            if (weight < 0.001f) { 
                if (rand01() > 0.1f)
                    break;
                weight /= 0.1f;
            }

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
        }
    }

    for (int i = 0; i < SHELLS; i++) {
        atomicAdd(&heat[i], _heat[i]);
    }
}

int main() {

    // histogram
    float * heat;
    checkCudaCall(cudaMallocManaged(&heat, SHELLS * sizeof(float)));

    // gpu timers
    cudaEvent_t gpu_start, gpu_finish; // timers
    checkCudaCall(cudaEventCreate(&gpu_start));
    checkCudaCall(cudaEventCreate(&gpu_finish));

    // kernel parameters
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_ceil(PHOTONS, BLOCK_SIZE * WORK));

    // launch kernel
    checkCudaCall(cudaEventRecord(gpu_start));
    photon<<<grid, block>>>(heat);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaEventRecord(gpu_finish));
    checkCudaCall(cudaDeviceSynchronize());

    // elapsed time
    float gpu_elapsed;
    checkCudaCall(cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_finish));
    checkCudaCall(cudaEventDestroy(gpu_start));
    checkCudaCall(cudaEventDestroy(gpu_finish));

    // heat2 calc
    float heat2[SHELL] = {0};
    for (int i = 0; i < SHELLS; i++) {
        heat2[i] = heat[i] * heat[i];
    }

    // output
#if VERBOSE
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
    printf("block size: %d\n", BLOCK_SIZE);
    printf("grid size: %d\n", div_ceil(PHOTONS, BLOCK_SIZE * WORK));
    printf("total threads: %d\n\n", BLOCK_SIZE * div_ceil(PHOTONS, BLOCK_SIZE * WORK);
    printf("%photons: lf\n", PHOTONS);
    printf("%ph/s: lf\n", 1e-3 * PHOTONS / gpu_elapsed);
    printf("gpu: %f ms\n", gpu_elapsed);
    // printf("total: %f ms\n", elapsed);
#else
    printf("block size: %d\n", BLOCK_SIZE);
    printf("grid size: %d\n", div_ceil(PHOTONS, BLOCK_SIZE * WORK));
    printf("total threads: %d\n\n", BLOCK_SIZE * div_ceil(PHOTONS, BLOCK_SIZE * WORK);
    printf("%photons: lf\n", PHOTONS);
    printf("%ph/s: lf\n", 1e-3 * PHOTONS / gpu_elapsed);
    printf("gpu: %f ms\n", gpu_elapsed);
    // printf("total: %f ms\n", elapsed);
#endif

    // cleanup
    checkCudaCall(cudaFree(heat));

    return 0;
}
