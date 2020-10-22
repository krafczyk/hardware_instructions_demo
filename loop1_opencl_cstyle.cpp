#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <unistd.h>

#include <CL/cl.h>

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
    size_t num_gen = 100000000;

    float* array1 = new float[num_gen];
    float* array2 = new float[num_gen];
    float* array3 = new float[num_gen];

    // Fill arrays with values
    for(size_t i=0; i < num_gen; ++i) {
        array1[i] = gen();
        array2[i] = gen();
    }

    // Initialize OpenCL Context and queue
    
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);

    // Create memory buffers
    cl_mem array1_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    cl_mem array2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, num_gen*sizeof(float), NULL, &ret);
    cl_mem array3_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_gen*sizeof(float), NULL, &ret);

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
