#include <iostream>
#include <random>
#include <functional>
#include <chrono>

inline float to_float(int in) {
    return (float) in;
}

void add_arrays(float* array1, float* array2, float* array3, size_t N) {
    for(size_t i=0; i < N; ++i) {
        array3[i] = array1[i]+array2[i];
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

    float* array1 = new float[num_gen];
    float* array2 = new float[num_gen];
    float* array3 = new float[num_gen];

    // Fill arrays with values
    for(size_t i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    auto start = std::chrono::high_resolution_clock::now();

    add_arrays(array1, array2, array3, num_gen);

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

    delete [] array1;
    delete [] array2;
    delete [] array3;

    return 0;
}
