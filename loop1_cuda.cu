#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <unistd.h>

#ifndef BUF_KIND
#define BUF_KIND 0
#endif

#if(BUF_KIND == 0)
#define BUF_TYPE float
#else
#define BUF_TYPE double
#endif

#define CudaWrap(EXP) \
{ \
    auto ret = EXP; \
    if (ret != cudaSuccess) { \
        std::cerr << "Error! " << cudaGetErrorString(ret) << " (" << ret << ")" << std::endl; \
        return 1; \
    }\
}

__global__
void addKernel(const BUF_TYPE* A, BUF_TYPE* B, size_t size) {
    size_t i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < size) {
        B[i] = A[i] + B[i];
    }
}

int main() {
    // Initialize random number generator
    std::mt19937_64 generator;
    generator.seed(42);
    std::uniform_real_distribution<BUF_TYPE> distribution(-1., 1.);
    auto gen = std::bind(distribution, generator);

    // Initialize arrays
    size_t num_gen = (1<<(30))/sizeof(BUF_TYPE);

    // Allocate arrays.
    BUF_TYPE* array1 = new BUF_TYPE[num_gen];
    BUF_TYPE* array2 = new BUF_TYPE[num_gen];

    // Allocate Device arrays.
    BUF_TYPE* dev_1 = nullptr;
    BUF_TYPE* dev_2 = nullptr;
    CudaWrap(cudaMalloc(&dev_1, num_gen*sizeof(BUF_TYPE)));
    CudaWrap(cudaMalloc(&dev_2, num_gen*sizeof(BUF_TYPE)));

    // Fill arrays with values
    for(size_t i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    // Copy data to Device
    cudaMemcpy(dev_1, array1, num_gen*sizeof(BUF_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_2, array2, num_gen*sizeof(BUF_TYPE), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    // Compute on Device
    int blockSize = 256;
    int numBlocks = (num_gen+blockSize-1)/(blockSize);
    addKernel<<<numBlocks,blockSize>>>(array1, array2, num_gen);
    // Copy data out of Device
    cudaMemcpy(array2, dev_2, num_gen*sizeof(BUF_TYPE), cudaMemcpyDeviceToHost);
    // Wait for operations to finish.
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();


    // Do something with the arrays so the addition isn't optimized out.
    BUF_TYPE sum = 0.;
    for(size_t i=0; i < num_gen; ++i) {
        sum += array2[i];
    }

    std::cout << sum << std::endl;
    std::cout << std::hexfloat;
    std::cout << sum << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

    std::cout << "Took " << duration.count() << " milliseconds" << std::endl;

    cudaFree(dev_1);
    cudaFree(dev_2);

    delete [] array1;
    delete [] array2;

    return 0;
}
