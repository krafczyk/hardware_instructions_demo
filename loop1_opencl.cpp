#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <unistd.h>

#include "opencl_helper.h"

inline float to_float(int in) {
    return (float) in;
}

std::string add_arrays_kernel =
    "__kernel void array_add(__global const float* A, __global const float* B, __global float* C) {"
    "   // Get the index of the current element to be processed"
    "   int i = get_global_id(0);"
    ""
    "   // Do the operation"
    "   C[i] = A[i] + B[i];"
    "}";

int main() {
    // Initialize random number generator
    std::mt19937_64 generator;
    generator.seed(42);
    std::uniform_real_distribution<float> distribution(-1., 1.);
    auto gen = std::bind(distribution, generator);

    // Initialize arrays
    int num_gen = 100000000;

    float* array1 = new float[num_gen];
    float* array2 = new float[num_gen];
    //float* array3 = new float[num_gen];

    // Fill arrays with values
    for(int i=0; i < num_gen; ++i) {
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

    // Get openCL ready
    //
    /*
    //
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    
    return 0;

    // Create memory buffers
    cl_mem array1_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    cl_mem array2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    cl_mem array3_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);

    // Create program for the kernel
    size_t prog_size = add_arrays_kernel.size();
    const char* program_str = add_arrays_kernel.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &program_str, &prog_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "array_add", &ret);

    auto start = std::chrono::high_resolution_clock::now();

    // Copy arrays to their buffers
    ret = clEnqueueWriteBuffer(command_queue, array1_mem_obj, CL_TRUE, 0, num_gen*sizeof(float), array1, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, array2_mem_obj, CL_TRUE, 0, num_gen*sizeof(float), array2, 0, NULL, NULL);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &array1_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &array2_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &array3_mem_obj);

    // Execute kernel
    size_t global_item_size = num_gen;
    size_t local_item_size = 64;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the result memory buffer out

    ret = clEnqueueReadBuffer(command_queue, array3_mem_obj, CL_TRUE, 0, num_gen*sizeof(float), array3, 0, NULL, NULL);

    auto stop = std::chrono::high_resolution_clock::now();

    // Do something with the arrays so the addition isn't optimized out.
    float sum = 0.;
    for(int i=0; i < num_gen; ++i) {
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
    */
}
