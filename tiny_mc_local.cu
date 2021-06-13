#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <curand_kernel.h>

#include "helper_cuda.h"
#include "wtime.h"
#include "params.h"

template <typename T>
T div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

__global__ void setup_prng(curandState * prng_states) {
  int id = threadIdx.x + blockIdx.x * blockDim.x; 

  /* Copy state to local memory for efficiency */ 
  curandState localState = prng_states[id]; 

  /* Each thread gets same seed, a different sequence
     number, no offset */
  curand_init(1234, id, 0, &prng_states[id]);
}

__global__ void photon(float * heat, curandState * prng_states) {
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    curandState prng_state = prng_states[id];

    int thread_from = WORK * (blockDim.x * blockIdx.x + threadIdx.x);
    int thread_to = min(WORK * (blockDim.x * blockIdx.x + threadIdx.x + 1),
            PHOTONS);

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

    for (int id = thread_from; id < thread_to; ++id)
    {
        for (;;) {
            /* move */
            float t = -logf(curand_uniform(prng_state));
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
                if (curand_uniform(prng_state) > 0.1f)
                    break;
                weight /= 0.1f;
            }

            /* New direction, rejection method */
            float xi1, xi2;
            do {
                xi1 = 2.0f * curand_uniform(prng_state) - 1.0f;
                xi2 = 2.0f * curand_uniform(prng_state) - 1.0f;
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
    const unsigned int threads_per_block = BLOCK_SIZE;
    const unsigned int block_count = div_ceil(PHOTONS, BLOCK_SIZE * WORK);
    const unsigned int total_threads = threads_per_block * block_count;

    // seed/PRNG setup
    curandState *prng_states;
    checkCudaCall((cudaMalloc((void **)&devStates,
                    totalThreads * sizeof(curandState))); 

    setup_prng<<<block_count, threads_per_block>>>(prng_states);
    checkCudaCall(cudaGetLastError());

    // launch kernel
    checkCudaCall(cudaEventRecord(gpu_start));
    photon<<<block_count, threads_per_block>>>(heat, prng_states);
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
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t /
               (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
#else // copiar lo que está abajo arriba cuando no haya mas cosas que agregar
    printf("block size: %u\n", BLOCK_SIZE);
    printf("grid size: %u\n", div_ceil(PHOTONS, BLOCK_SIZE * WORK));
    printf("total threads: %u\n", BLOCK_SIZE * 
            div_ceil(PHOTONS, BLOCK_SIZE * WORK);
    printf("device allocated : % lu\n", sizeof(curandState) * total_threads);
    printf("%photons: lf\n", PHOTONS);
    printf("%ph/s: lf\n", 1e-3 * PHOTONS / gpu_elapsed);
    printf("gpu: %f ms\n", gpu_elapsed);
    // printf("total: %f ms\n", elapsed);
#endif

    // cleanup
    checkCudaCall(cudaFree(heat));
    checkCudaCall(cudaFree(prng_states));

    return 0;
}
