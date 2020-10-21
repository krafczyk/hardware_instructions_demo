#include <random>

int main() {

    // Initialize random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1., 1.);
    auto gen = std::bind(distribution, generator);

    // Initialize arrays
    int num_gen = 100000;

    float array1[num_gen];
    float array2[num_gen];
    float array3[num_gen];

    // Fill arrays with values
    for(int i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    // Add arrays together (This should be vectorized)
    for(int i=0; i < num_gen; ++i) {
        array3[i] = array1[i]+array2[i];
    }

    // Do something with the arrays so the addition isn't optimized out.
    float sum = 0.;
    for(int i=0; i < num_gen; ++i) {
        sum += array3[i];
    }

    return 0;
}
