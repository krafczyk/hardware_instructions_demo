#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "ArgParseStandalone.h"

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

inline size_t Idx(const size_t x, const size_t y, const size_t len) {
    return x+(y*len);
}

__device__
inline size_t IdxDevice(const size_t x, const size_t y, const size_t len) {
    return x+(y*len);
}

__global__
void SimKernel(BUF_TYPE* in_buf, BUF_TYPE* out_buf, const size_t side_length, const BUF_TYPE alpha, const BUF_TYPE dx) {
    const size_t i = blockIdx.x*blockDim.x+threadIdx.x;
    const size_t y_i = i/side_length;
    const size_t x_i = i-(y_i*side_length);
    if ((x_i > 0)&&(x_i < side_length-1)&&(y_i > 0)&&(y_i < side_length-1)) {
        const BUF_TYPE up = in_buf[IdxDevice(x_i, y_i-1, side_length)];
        const BUF_TYPE down = in_buf[IdxDevice(x_i, y_i+1, side_length)];
        const BUF_TYPE left = in_buf[IdxDevice(x_i-1, y_i, side_length)];
        const BUF_TYPE right = in_buf[IdxDevice(x_i+1, y_i, side_length)];
        const BUF_TYPE center = in_buf[i];
        const BUF_TYPE lap = (up+down+left+right-4*center)/(dx*dx);
        const BUF_TYPE tdiff = alpha*lap;
        // Compute update
        out_buf[i] = in_buf[i]+tdiff;
    }
}

int main(int argc, char** argv) {
    const size_t mb_size = 1 << 20;
    size_t target_buf_size = 1 << 7;

    // Alpha constant
    BUF_TYPE alpha = 0.01;
    // Time spacing
    BUF_TYPE dt = 0.1;
    // Spatial spacing
    BUF_TYPE dx = 1.;
    // Number of steps
    size_t N = 10;

    ArgParse::ArgParser Parser("Laplace simulation CPU");
    Parser.AddArgument("-size/--buf-size", "Target size of buffers in MB.", &target_buf_size);

    if (Parser.ParseArgs(argc, argv) != 0) {
        std::cout << "Problem parsing arguments!" << std::endl;
        return 1;
    }

    if (Parser.HelpPrinted()) {
        return 0;
    }

    std::cout << "Simulation Parameters" << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "dx: " << dx << std::endl;
    std::cout << "Number of Timestep: " << N << std::endl;

    // Compute rough 'edge' size.
    size_t total_buf_size = target_buf_size*mb_size;
    size_t total_possible_elements = total_buf_size/sizeof(BUF_TYPE);

    size_t side_length = std::floor(std::pow((BUF_TYPE)total_possible_elements, 1./2.));
    std::cout << "Full side length: " << side_length << std::endl;
    size_t lattice_length = side_length-2;
    std::cout << "lattice side length: " << lattice_length << std::endl;
    size_t total_length = side_length*side_length;
    std::cout << "Total elements: " << total_length << std::endl;
    std::cout << "Size of one buffer: " << total_length*sizeof(BUF_TYPE) << " bytes." << std::endl;

    // Allocate and Initialize buffers
    BUF_TYPE* t1 = new BUF_TYPE[total_length];
    for (size_t i = 0; i < total_length; ++i) {
        t1[i] = 0.;
    }
    // Initial conditions May not be directly in the middle.
    t1[Idx(side_length/2,side_length/2, side_length)] = 100.;       
 
    BUF_TYPE* dev_1 = nullptr;
    BUF_TYPE* dev_2 = nullptr;
    CudaWrap(cudaMalloc(&dev_1, total_length*sizeof(BUF_TYPE)));
    CudaWrap(cudaMalloc(&dev_2, total_length*sizeof(BUF_TYPE)));
    CudaWrap(cudaMemcpy(dev_1, t1, total_length*sizeof(BUF_TYPE), cudaMemcpyHostToDevice));

    std::cout << "Start simulation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    BUF_TYPE t = 0.;
    for(size_t t_i=0; t_i < N; ++t_i) {
        // We don't go from beginning to end of array to simplify logic.
        int blockSize = 256;
        int numBlocks = (total_length+blockSize-1)/(blockSize);
        SimKernel<<<numBlocks,blockSize>>>(dev_1, dev_2, side_length, alpha, dx);

        // Swap buffers
        BUF_TYPE* dev_temp = dev_1;
        dev_1 = dev_2;
        dev_2 = dev_temp;

        // Iterate time
        t += dt;
    }
    
    // Copy result out
    CudaWrap(cudaDeviceSynchronize());
    CudaWrap(cudaMemcpy(t1, dev_1, total_length*sizeof(BUF_TYPE), cudaMemcpyDeviceToHost));

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Simulation Finished." << std::endl;

    // Now we have the result.
    BUF_TYPE buf_max = 0.;
    BUF_TYPE avg = 0.;
    for(size_t x_i = 1; x_i < side_length-1; ++x_i) {
        for(size_t y_i = 1; y_i < side_length-1; ++y_i) {
            BUF_TYPE val = t1[Idx(x_i, y_i, side_length)];
            buf_max = std::max(val, buf_max);
            avg += val;
        }
    }
    avg = avg/((BUF_TYPE)(lattice_length*lattice_length));

    cudaFree(dev_1);
    cudaFree(dev_2);

    delete [] t1;

    std::cout << "Number of steps taken: " << N << std::endl;
    std::cout << "Final time: " << t << std::endl;
    std::cout << "Max value: " << buf_max << std::endl;
    std::cout << "Average value: " << avg << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    std::cout << "Took " << (duration.count()/((BUF_TYPE)N)) << " milliseconds per iteration" << std::endl;
    return 0;
}
