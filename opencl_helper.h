#ifndef OPENCL_HELPER_HDR
#define OPENCL_HELPER_HDR

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>

std::vector<cl::Device> get_devices(const cl_device_type dev_type) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    std::vector<cl::Device> all_gpu_devices;
    for (auto& p: all_platforms) {
        std::vector<cl::Device> gpu_devices;
        cl_int ret = p.getDevices(dev_type, &gpu_devices);
        if (ret == CL_SUCCESS) {
            all_gpu_devices.insert(all_gpu_devices.end(), gpu_devices.begin(), gpu_devices.end());
        }
    }
    return all_gpu_devices;
}

#endif
