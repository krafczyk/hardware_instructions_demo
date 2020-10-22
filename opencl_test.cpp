#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {
    cl_int ret = 0;

    // Get number of platforms
    cl_uint num_platforms = 0;
    ret = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ret != CL_SUCCESS) {
        std::cerr << "There was an error querying available platforms" << std::endl;
        return 1;
    }

    std::cout << "There are " << num_platforms << " Platforms" << std::endl;

    // Get platform ids
    cl_platform_id platforms[num_platforms];
    ret = clGetPlatformIDs(num_platforms, platforms, &num_platforms);

    for (int i = 0; i < num_platforms; ++i) {
        cl_platform_id platform_id = platforms[i];
        std::cout << "Platform: " << platforms[i] << std::endl;
        cl_uint num_devices = 0;
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (ret != CL_SUCCESS) {
            std::cerr << "There was a problem listing devices on platform " << platform_id << std::endl;
        }
        cl_device_id devices[num_devices];
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
        std::cout << "Devices:" << std::endl;
        for (int j = 0; j < num_devices; ++j) {
            cl_device_id device_id = devices[j];
            cl_device_type dev_type;
            ret = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
            if (ret != CL_SUCCESS) {
                std::cerr << "There was a problem getting device type information!" << std::endl;
            }
            const size_t device_name_len = 2048;
            char device_name[device_name_len];
            size_t name_len = 0;
            ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_len*sizeof(char), device_name, &name_len);
            if (ret != CL_SUCCESS) {
                std::cerr << "There was a problem getting device name information!" << std::endl;
            }
            cl_uint global_cacheline_size = 0;
            ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &global_cacheline_size, NULL);
            if (ret != CL_SUCCESS) {
                std::cerr << "There was a problem getting device cacheline information!" << std::endl;
            }
            std::cout << device_name << "(" << device_id << "): Device Type: ";
            if (dev_type & CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU";
            }
            if (dev_type & CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU";
                cl_uint pci_bus_id = 0;
                const cl_device_info CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
                ret = clGetDeviceInfo(device_id, CL_DEVICE_PCI_BUS_ID_NV, sizeof(cl_uint), &pci_bus_id, NULL);
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
            if (dev_type & CL_DEVICE_TYPE_CUSTOM) {
                std::cout << "CUSTOM";
            }
            std::cout << " cacheline size: " << global_cacheline_size;
            std::cout << std::endl;
        }
    }

    return 0;
}
