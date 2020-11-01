#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <unistd.h>

#include "opencl_helper.h"

#ifndef BUF_KIND
#define BUF_KIND 0
#endif

#if(BUF_KIND == 0)
#define BUF_TYPE float
#else
#define BUF_TYPE double
#endif

inline float to_float(int in) {
    return (float) in;
}

std::string add_arrays_kernel = R"CODE(__kernel void array_add(__global const float* A, __global const float* B, __global float* C) {
    // Get the index of the current element to be processed
    int i = get_global_id(0);

    // Do the operation
    C[i] = A[i] + B[i];
})CODE";

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

    cl::Device device_to_use;
    bool set_device = false;

    auto gpu_devices = get_devices(CL_DEVICE_TYPE_GPU);
    if (gpu_devices.size() > 1) {
        if (gpu_devices.size() == 1) {
            device_to_use = gpu_devices[0];
        }
        const size_t hostname_len = 1024;
        char hostname[hostname_len];
        gethostname(hostname, hostname_len);
        if(std::string(hostname) == "schumann") {
            for (auto& d: gpu_devices) {
                cl_int pci_bus_id = 0;
                const cl_device_info CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
                cl_int ret = 0;
                ret = d.getInfo(CL_DEVICE_PCI_BUS_ID_NV, &pci_bus_id);
                if (ret != CL_SUCCESS) {
                    std::cerr << "There was a problem getting the GPU BUS Id!" << std::endl;
                    return 1;
                }
                if (pci_bus_id == 3) {
                    device_to_use = d;
                    set_device = true;
                }
            }
        } else {
            device_to_use = gpu_devices[0];
            set_device = true;
        }
    }
    if (set_device) {
        std::cout << "Selected GPU as device" << std::endl;
    } else {
        auto cpu_devices = get_devices(CL_DEVICE_TYPE_CPU);
        if (cpu_devices.size() > 0) {
            device_to_use = cpu_devices[0];
            set_device = true;
        }
        if (!set_device) {
            std::cerr << "No supported devices are available." << std::endl;
        } else {
            std::cout << "Selected CPU as device" << std::endl;
        }
    }

    // Create context
    std::vector<cl::Device> device_container;
    device_container.push_back(device_to_use);

    cl::Context context(device_container);

    // Build program on our context
    cl::Program addArraysProgram(context, add_arrays_kernel);
    try {
        cl_int ret = addArraysProgram.build();
        if(ret != CL_SUCCESS) {
            std::cerr << "Problem building OpenCL Program!" << std::endl;
            return 1;
        }
    } catch (...) {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = addArraysProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        std::cerr << "Problem building OpenCL Program!" << std::endl;
        for (auto& pair: buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    cl_int ret = 0;
    cl::Buffer array1_buf_obj(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error allocating buffer!" << std::endl;
    }
    cl::Buffer array2_buf_obj(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error allocating buffer!" << std::endl;
    }
    cl::Buffer array3_buf_obj(context, CL_MEM_WRITE_ONLY, num_gen*sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error allocating buffer!" << std::endl;
    }

    cl::CommandQueue command_queue(context);

    cl::Kernel kernel(addArraysProgram, "add_arrays_kernel");
    kernel.setArg(0, (void*) &array1_buf_obj);
    kernel.setArg(1, (void*) &array2_buf_obj);
    kernel.setArg(2, (void*) &array3_buf_obj);

    cl::NDRange global_range(num_gen);
    cl::NDRange local_range(64);
    auto start = std::chrono::high_resolution_clock::now();
    command_queue.enqueueWriteBuffer(array1_buf_obj, CL_TRUE, 0, num_gen*sizeof(float), array1);
    command_queue.enqueueWriteBuffer(array2_buf_obj, CL_TRUE, 0, num_gen*sizeof(float), array2);
    command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
    command_queue.enqueueReadBuffer(array3_buf_obj, CL_TRUE, 0, num_gen*sizeof(float), array3);
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
