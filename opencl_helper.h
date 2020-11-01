#ifndef OPENCL_HELPER_HDR
#define OPENCL_HELPER_HDR

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <map>
#include "unistd.h"

#define CaseReturnString(x) case x: return #x;

const char *opencl_errstr(cl_int err)
{
    switch (err)
    {
        CaseReturnString(CL_SUCCESS                        )                                  
        CaseReturnString(CL_DEVICE_NOT_FOUND               )
        CaseReturnString(CL_DEVICE_NOT_AVAILABLE           )
        CaseReturnString(CL_COMPILER_NOT_AVAILABLE         ) 
        CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        CaseReturnString(CL_OUT_OF_RESOURCES               )
        CaseReturnString(CL_OUT_OF_HOST_MEMORY             )
        CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE   )
        CaseReturnString(CL_MEM_COPY_OVERLAP               )
        CaseReturnString(CL_IMAGE_FORMAT_MISMATCH          )
        CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        CaseReturnString(CL_BUILD_PROGRAM_FAILURE          )
        CaseReturnString(CL_MAP_FAILURE                    )
        CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        CaseReturnString(CL_COMPILE_PROGRAM_FAILURE        )
        CaseReturnString(CL_LINKER_NOT_AVAILABLE           )
        CaseReturnString(CL_LINK_PROGRAM_FAILURE           )
        CaseReturnString(CL_DEVICE_PARTITION_FAILED        )
        CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        CaseReturnString(CL_INVALID_VALUE                  )
        CaseReturnString(CL_INVALID_DEVICE_TYPE            )
        CaseReturnString(CL_INVALID_PLATFORM               )
        CaseReturnString(CL_INVALID_DEVICE                 )
        CaseReturnString(CL_INVALID_CONTEXT                )
        CaseReturnString(CL_INVALID_QUEUE_PROPERTIES       )
        CaseReturnString(CL_INVALID_COMMAND_QUEUE          )
        CaseReturnString(CL_INVALID_HOST_PTR               )
        CaseReturnString(CL_INVALID_MEM_OBJECT             )
        CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CaseReturnString(CL_INVALID_IMAGE_SIZE             )
        CaseReturnString(CL_INVALID_SAMPLER                )
        CaseReturnString(CL_INVALID_BINARY                 )
        CaseReturnString(CL_INVALID_BUILD_OPTIONS          )
        CaseReturnString(CL_INVALID_PROGRAM                )
        CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE     )
        CaseReturnString(CL_INVALID_KERNEL_NAME            )
        CaseReturnString(CL_INVALID_KERNEL_DEFINITION      )
        CaseReturnString(CL_INVALID_KERNEL                 )
        CaseReturnString(CL_INVALID_ARG_INDEX              )
        CaseReturnString(CL_INVALID_ARG_VALUE              )
        CaseReturnString(CL_INVALID_ARG_SIZE               )
        CaseReturnString(CL_INVALID_KERNEL_ARGS            )
        CaseReturnString(CL_INVALID_WORK_DIMENSION         )
        CaseReturnString(CL_INVALID_WORK_GROUP_SIZE        )
        CaseReturnString(CL_INVALID_WORK_ITEM_SIZE         )
        CaseReturnString(CL_INVALID_GLOBAL_OFFSET          )
        CaseReturnString(CL_INVALID_EVENT_WAIT_LIST        )
        CaseReturnString(CL_INVALID_EVENT                  )
        CaseReturnString(CL_INVALID_OPERATION              )
        CaseReturnString(CL_INVALID_GL_OBJECT              )
        CaseReturnString(CL_INVALID_BUFFER_SIZE            )
        CaseReturnString(CL_INVALID_MIP_LEVEL              )
        CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE       )
        CaseReturnString(CL_INVALID_PROPERTY               )
        CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR       )
        CaseReturnString(CL_INVALID_COMPILER_OPTIONS       )
        CaseReturnString(CL_INVALID_LINKER_OPTIONS         )
        CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT )
        default: return "Unknown OpenCL error code";
	}
}

std::vector<cl_device_id> get_devices(const cl_device_type dev_type, cl_int* ret) {
	std::vector<cl_device_id> devices;
	cl_uint num_platforms = 0;
	(*ret) = clGetPlatformIDs(0, NULL, &num_platforms);
	if (*ret != CL_SUCCESS) {
		std::cerr << "Error getting number of platforms!" << std::endl;
		return devices;
	}
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	(*ret) = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (*ret != CL_SUCCESS) {
		std::cerr << "Error getting platform Ids!" << std::endl;
		delete [] platforms;
		return devices;
	}
	for (cl_uint plat_i = 0 ; plat_i < num_platforms; ++plat_i) {
		auto platform_id = platforms[plat_i];
		cl_uint num_devices = 0;
		(*ret) = clGetDeviceIDs(platform_id, dev_type, 0, NULL, &num_devices);
		if (*ret == CL_DEVICE_NOT_FOUND) {
			// Didin't find any devices of the specified type.
			continue;
		}
		if (*ret != CL_SUCCESS) {
			std::cerr << "Error getting number of devices on platform" << std::endl;
			delete [] platforms;
			return devices;
		}
		cl_device_id* device_list = new cl_device_id[num_devices];
		(*ret) = clGetDeviceIDs(platform_id, dev_type, num_devices, device_list, NULL);
		if (*ret != CL_SUCCESS) {
			std::cerr << "Error getting devices on platform" << std::endl;
			delete [] platforms;
			delete [] device_list;
			return devices;
		}
		for(cl_uint dev_i = 0; dev_i < num_devices; ++dev_i) {
			auto device_id = device_list[dev_i];
			devices.push_back(device_id);
		}
		delete [] device_list;
	}
	delete [] platforms;

	*ret = CL_SUCCESS;
	return devices;
}

std::vector<cl_device_id> get_devices_full(bool gpu, cl_int* ret) {
	std::vector<cl_device_id> device_container;
	if (gpu) {
		auto devices = get_devices(CL_DEVICE_TYPE_GPU, ret);
		if (*ret != CL_SUCCESS) {
			std::cerr << "Problem getting GPU devices!" << std::endl;
			return device_container;
		} else {
			const size_t hostname_len = 1024;
        	char hostname[hostname_len];
			gethostname(hostname, hostname_len);
        	if(std::string(hostname) == "schumann") {
            	for (auto& d: devices) {
                	cl_int pci_bus_id = 0;
                	const cl_device_info CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
					(*ret) = clGetDeviceInfo(d, CL_DEVICE_PCI_BUS_ID_NV, sizeof(cl_int), &pci_bus_id, NULL);
                	if (ret != CL_SUCCESS) {
                    	std::cerr << "There was a problem getting the GPU BUS Id!" << std::endl;
						std::cerr << "ahhh" << std::endl;
						std::cerr << opencl_errstr(*ret) << std::endl;
                    	return device_container;
                	}
                	if (pci_bus_id == 3) {
						device_container.push_back(d);
						return device_container;
                	}
            	}
				device_container.push_back(devices[0]);
				return device_container;
        	} else {
				device_container.push_back(devices[0]);
				return device_container;
        	}
		}
		std::cerr << "There was a problem finding a GPU!" << std::endl;
		return device_container;
	} else {
		auto devices = get_devices(CL_DEVICE_TYPE_CPU, ret);
		if (*ret != CL_SUCCESS) {
			std::cerr << "Problem getting CPU devices!" << std::endl;
			return device_container;
		} else {
			device_container.push_back(devices[0]);
			return device_container;
		}
	}
}

typedef std::pair<cl_platform_id,std::vector<cl_device_id>> platform_device_pair_t;
platform_device_pair_t get_devices_and_platform(cl_device_type dev_type, bool all_devices, cl_int* ret) {
	platform_device_pair_t result;
	std::vector<cl_device_id> device_container;
	// First, get platforms.
	std::map<cl_platform_id, cl_uint> number_of_appropriate_devices;
	cl_uint num_platforms = 0;
	(*ret) = clGetPlatformIDs(0, NULL, &num_platforms);
	if (*ret != CL_SUCCESS) {
		std::cerr << "Problem getting number of platforms." << std::endl;
		return result;
	}

	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	(*ret) = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (*ret != CL_SUCCESS) {
		std::cerr << "Problem getting the platforms." << std::endl;
		return result;
	}
	for (cl_uint plat_i = 0; plat_i < num_platforms; ++plat_i) {
		// Get number of devices on each platform.
		auto platform = platforms[plat_i];
		cl_uint num_devices = 0;
		(*ret) = clGetDeviceIDs(platform, dev_type, 0, NULL, &num_devices);
		if (*ret == CL_SUCCESS) {
			number_of_appropriate_devices[platform] = num_devices;
		} else if (*ret == CL_DEVICE_NOT_FOUND) {
			number_of_appropriate_devices[platform] = 0;
		} else {
			std::cerr << "Problem getting a device listing." << std::endl;
			return result;
		}
	}
	// Cleanup after ourselves
	delete [] platforms;

	cl_platform_id device_platform = 0;
	cl_uint num_devices = 0;
	for(auto element: number_of_appropriate_devices) {
		if (element.second > num_devices) {
			device_platform = element.first;
			num_devices = element.second;
		}
	}

	if (num_devices == 0) {
		std::cerr << "No devices for some reason!" << std::endl;
		return result;
	}

	// Get devices for the platform
	std::vector<cl_device_id> platform_devices;
	cl_device_id* devices = new cl_device_id[num_devices];
	*ret = clGetDeviceIDs(device_platform, dev_type, num_devices, devices, 0);
	if(*ret != CL_SUCCESS) {
		std::cerr << "There was a problem listing devices" << std::endl;
		delete [] devices;
		return result;
	}
	for(size_t dev_i = 0; dev_i < num_devices; ++dev_i) {
		auto device = devices[dev_i];
		platform_devices.push_back(device);
	}
	delete [] devices;

	if (all_devices) {
		return platform_device_pair_t(device_platform, platform_devices);
	} else {
		std::vector<cl_device_id> device_container;
		if (dev_type == CL_DEVICE_TYPE_GPU) {
			const size_t hostname_len = 1024;
        	char hostname[hostname_len];
			gethostname(hostname, hostname_len);
        	if(std::string(hostname) == "schumann") {
            	for (auto& d: platform_devices) {
                	cl_int pci_bus_id = 0;
                	const cl_device_info CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
					(*ret) = clGetDeviceInfo(d, CL_DEVICE_PCI_BUS_ID_NV, sizeof(cl_int), &pci_bus_id, NULL);
                	if (*ret != CL_SUCCESS) {
                    	std::cerr << "There was a problem getting the GPU BUS Id!" << std::endl;
						std::cerr << opencl_errstr(*ret) << std::endl;
                    	return result;
                	}
                	if (pci_bus_id == 3) {
						device_container.push_back(d);
						return platform_device_pair_t(device_platform, device_container);
                	}
            	}
				device_container.push_back(devices[0]);
				return platform_device_pair_t(device_platform, device_container);
			}
		}
		// Otherwise, grab first device.
		device_container.push_back(platform_devices[0]);
		return platform_device_pair_t(device_platform, device_container);
	}
}

#endif
