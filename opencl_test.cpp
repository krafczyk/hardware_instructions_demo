#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include "opencl_helper.h"

int main() {

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    std::cout << "There are " << all_platforms.size() << " platforms" << std::endl;
    for (auto& p: all_platforms) {
        std::cout << "Platform " << p() << std::endl;

        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices); 

        for(auto& d: devices) {
            cl_device_type dev_type = 0;
            cl_uint ret = d.getInfo(CL_DEVICE_TYPE, &dev_type);
            if (ret != CL_SUCCESS) {
                std::cerr << "Problem getting device type" << std::endl;
                return 1;
            }
            std::string dev_name;
            ret = d.getInfo(CL_DEVICE_NAME, &dev_name);
            if (ret != CL_SUCCESS) {
                std::cerr << "Problem getting device name" << std::endl;
            }
            cl_uint global_cacheline_size = 0;
            ret = d.getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &global_cacheline_size);
            if (ret != CL_SUCCESS) {
                std::cerr << "There was a problem getting device cacheline information!" << std::endl;
            }
            std::cout << "Device " << dev_name << "(" << d() << ") ";
            if (dev_type & CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU";
            }
            if (dev_type & CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU";
                cl_uint pci_bus_id = 0;
                const cl_device_info CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
                ret = d.getInfo(CL_DEVICE_PCI_BUS_ID_NV, &pci_bus_id);
                if (ret != CL_SUCCESS) {
                    std::cerr << "There was a problem getting the GPU PCI BUS ID!" << std::endl;
                    return 1;
                }
                std::cout << " (" << pci_bus_id << ")";
            }
            if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << "ACCELERATOR";
            }
            if (dev_type & CL_DEVICE_TYPE_DEFAULT) {
                std::cout << "DEFAULT";
            }
            std::cout << " cacheline size: " << global_cacheline_size;
            std::cout << std::endl;
        }
    }

    std::vector<cl::Device> gpu_devices = get_devices(CL_DEVICE_TYPE_GPU);

    std::cout << "GPUs are: " << std::endl;
    for (auto& d: gpu_devices) {
        std::cout << d() << std::endl;
    }

    return 0;
}
