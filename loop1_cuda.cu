#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <unistd.h>

inline float to_float(int in) {
    return (float) in;
}

__global__
void addKernel(const float* A, const float* B, float* C, size_t size) {
    size_t i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Initialize random number generator
    std::mt19937_64 generator;
    generator.seed(42);
    std::uniform_real_distribution<float> distribution(-1., 1.);
    auto gen = std::bind(distribution, generator);

    // Initialize arrays
    size_t num_gen = 1<<(30-2);

    float* array1;
    float* array2;
    float* array3;

    cudaMallocManaged(&array1, num_gen*sizeof(float));
    cudaMallocManaged(&array2, num_gen*sizeof(float));
    cudaMallocManaged(&array3, num_gen*sizeof(float));

    // Fill arrays with values
    for(size_t i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    auto start = std::chrono::high_resolution_clock::now();

    addKernel<<<1,1>>>(array1, array2, array3, num_gen);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    // Do something with the arrays so the addition isn't optimized out.
    float sum = 0.;
    for(size_t i=0; i < num_gen; ++i) {
        sum += array3[i];
    }

    std::cout << sum << std::endl;
    std::cout << std::hexfloat;
    std::cout << sum << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

    std::cout << "Took " << duration.count() << " milliseconds" << std::endl;

    cudaFree(array1);
    cudaFree(array2);
    cudaFree(array3);

    return 0;
}
