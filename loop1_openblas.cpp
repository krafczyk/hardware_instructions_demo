#include <iostream>
#include <random>
#include <functional>
#include <chrono>

#ifndef BUF_KIND
#define BUF_KIND 0
#endif

#if(BUF_KIND == 0)
#define BUF_TYPE float
#else
#define BUF_TYPE double
#endif

#include "f77blas.h"

void add_arrays(BUF_TYPE* array1, BUF_TYPE* array2, size_t N) {
    BUF_TYPE alpha = 1.0;
    blasint inc = 1;
    #if(BUF_KIND == 0)
    saxpy_((blasint*)&N, &alpha, array1, &inc, array2, &inc);
    #else
    daxpy_((blasint*)&N, &alpha, array1, &inc, array2, &inc);
    #endif
}

int main() {
    // Initialize random number generator
    std::mt19937_64 generator;
    generator.seed(42);
    std::uniform_real_distribution<BUF_TYPE> distribution(-1., 1.);
    auto gen = std::bind(distribution, generator);

    // Initialize arrays
    size_t num_gen = (1<<30)/sizeof(BUF_TYPE);

    std::cout << "Performing test with " << num_gen << " array elements" << std::endl;
    std::cout << "Array size: " << num_gen*sizeof(BUF_TYPE) << " Bytes" << std::endl;

    BUF_TYPE* array1 = new BUF_TYPE[num_gen];
    BUF_TYPE* array2 = new BUF_TYPE[num_gen];

    // Fill arrays with values
    for(size_t i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    auto start = std::chrono::high_resolution_clock::now();

    add_arrays(array1, array2, num_gen);

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

    delete [] array1;
    delete [] array2;

    return 0;
}
