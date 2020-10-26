#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <typeinfo>
#include <unistd.h>

#include "opencl_helper.h"

#include "ArgParseStandalone.h"

#ifndef BUF_TYPE
#define BUF_TYPE float
#endif

inline size_t Idx(const size_t x, const size_t y, const size_t len) {
    return x+(y*len);
}

std::string produce_sim_kernel_code(const size_t side_length, const BUF_TYPE alpha, const BUF_TYPE dx) {
    std::stringstream ss;
    std::string buf_type_string = typeid(BUF_TYPE).name();
    if (buf_type_string == "f") {
        buf_type_string = "float";
    } else if (buf_type_string == "d") {
        buf_type_string = "double";
    }
    ss << "__kernel void sim_kernel(__global const " << buf_type_string << "* in_buf, __global " << buf_type_string << "* out_buf) {" << std::endl;
    ss << "    const size_t i = get_global_id(0);" << std::endl;
    ss << "    const size_t y_i = i/" << side_length << ";" << std::endl;
    ss << "    const size_t x_i = i-(y_i*" << side_length << ");" << std::endl;
    ss << "    size_t I = 0;" << std::endl;
    ss << "    if ((x_i > 0)&&(x_i < " << (side_length-1) << ")&&(y_i > 0)&&(y_i < " << (side_length-1) << ")) {" << std::endl;
    ss << "        I = x_i+((y_i-1)*" << side_length << ");" << std::endl;
    ss << "        const " << buf_type_string << " up = in_buf[I];" << std::endl;
    ss << "        I = x_i+((y_i+1)*" << side_length << ");" << std::endl;
    ss << "        const " << buf_type_string << " down = in_buf[I];" << std::endl;
    ss << "        I = (x_i-1)+(y_i*" << side_length << ");" << std::endl;
    ss << "        const " << buf_type_string << " left = in_buf[I];" << std::endl;
    ss << "        I = (x_i+1)+(y_i*" << side_length << ");" << std::endl;
    ss << "        const " << buf_type_string << " right = in_buf[I];" << std::endl;
    ss << "        const " << buf_type_string << " center = in_buf[i];" << std::endl;
    ss.precision(std::numeric_limits<BUF_TYPE>::max_digits10);
    ss << "        const " << buf_type_string << " lap = (up+down+left+right-4*center)/" << (dx*dx) << ";" << std::endl;
    ss << "        const " << buf_type_string << " tdiff = " << alpha << "*lap;" << std::endl;
    ss << "        out_buf[i] = in_buf[i] + tdiff;" << std::endl;
    ss << "    }" << std::endl;
    ss << "}" << std::endl;
    return ss.str();
}

int main(int argc, char** argv) {
    const size_t mb_size = 1 << 20;
    //size_t target_buf_size = 1 << 10;
    size_t target_buf_size = 1 << 7;

    // Alpha constant
    BUF_TYPE alpha = 0.01;
    // Time spacing
    BUF_TYPE dt = 0.1;
    // Spatial spacing
    BUF_TYPE dx = 1.;
    // Number of steps
    size_t N = 10;

    bool gpu = false;
	bool all = false;

    ArgParse::ArgParser Parser("Laplace simulation CPU");
    Parser.AddArgument("-size/--buf-size", "Target size of buffers in MB.", &target_buf_size);
    Parser.AddArgument("-gpu", "Indicate you'd like to use a GPU, otherwise it'll use a cpu.", &gpu);
	Parser.AddArgument("-all", "Indicate you'd like to use all devices in the selected platform.", &all);

    if (Parser.ParseArgs(argc, argv) != 0) {
        std::cout << "Problem parsing arguments!" << std::endl;
        return 1;
    }

    if (Parser.HelpPrinted()) {
        return 0;
    }

	if (all) {
		std::cerr << "Using all devices on a platform is currently not supported." << std::endl;
		return 1;
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

    // Produce kernel program
    std::string sim_kernel_program_code = produce_sim_kernel_code(side_length, alpha, dx);

    //std::cout << "Program Code:" << std::endl;
    //std::cout << sim_kernel_program_code << std::endl;

	cl_int ret = 0;
	platform_device_pair_t pd;
	if (gpu) {
		pd = get_devices_and_platform(CL_DEVICE_TYPE_GPU, all, &ret);
	} else {
		pd = get_devices_and_platform(CL_DEVICE_TYPE_CPU, all, &ret);
	}
	if (ret != CL_SUCCESS) {
		std::cerr << "Error getting devices." << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		return 1;
	}
	if (pd.second.size() == 0) {
		std::cerr << "Didn't find any devices matching request!" << std::endl;
		return 1;
	}

    std::cout << "Device and Platform chosen" << std::endl;

    std::cout << "Create Context" << std::endl;
	cl_context_properties props[3];
	props[0] = CL_CONTEXT_PLATFORM;
	props[1] = (cl_context_properties) pd.first;
	props[2] = 0;
	cl_context context = clCreateContext(props, pd.second.size(), pd.second.data(), NULL, NULL, &ret);
	if(ret != CL_SUCCESS) {
		std::cerr << "Problem creating context!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		return 1;
	}	
    std::cout << "Context created" << std::endl;

	std::cout << "Create program object" << std::endl;
	const char* source = sim_kernel_program_code.c_str();
	cl_program sim_kernel_program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
	if (ret != CL_SUCCESS) {
		std::cerr << "Problem building code!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		return 1;
	}

    std::cout << "Start program build" << std::endl;
	ret = clBuildProgram(sim_kernel_program, pd.second.size(), pd.second.data(), NULL, NULL, NULL);
	if(ret != CL_SUCCESS) {
		std::cerr << "There was a problem building the program!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		const size_t log_len = 1024;
		char log[log_len];
		ret = clGetProgramBuildInfo(sim_kernel_program, pd.second[0], CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
		if (ret != CL_SUCCESS) {
			std::cerr << "Couldn't get build log!" << std::endl;
			std::cerr << opencl_errstr(ret) << std::endl;
			return 1;
		}
		std::cerr << "Build Log: " << std::endl;
		std::cerr << log << std::endl;
		return 1;
	}
    std::cout << "Program Built" << std::endl;

    BUF_TYPE* t1 = nullptr;

    // Initializing buffer
    total_length = side_length*side_length;
    t1 = new BUF_TYPE[side_length*side_length];
    for (size_t i = 0; i < total_length; ++i) {
        t1[i] = 0.;
    }
    // Initial conditions May not be directly in the middle.
    t1[Idx(side_length/2,side_length/2, side_length)] = 100.;       

    // Allocate buffers
    cl_mem t1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, total_length*sizeof(BUF_TYPE), NULL, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error allocating buffer!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
        return 1;
    }
    __attribute__((unused)) cl_mem t2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, total_length*sizeof(BUF_TYPE), NULL, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error allocating buffer!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
        return 1;
    }

    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, pd.second[0], NULL, &ret);
	if (ret != CL_SUCCESS) {
		std::cerr << "Error creating command queue" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		return 1;
	}

    ret = clEnqueueWriteBuffer(command_queue, t1_buf, CL_TRUE, 0, total_length*sizeof(BUF_TYPE), t1, 0, NULL, NULL);
    if(ret != CL_SUCCESS) {
        std::cerr << "Error queue write buffer" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
        return 1;
    }

    cl_kernel sim_kernel = clCreateKernel(sim_kernel_program, "sim_kernel", &ret);
	if(ret != CL_SUCCESS) {
		std::cerr << "Error creating kernel" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
		return 1;
	}

    std::cout << "Start simulation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    BUF_TYPE t = 0.;
	size_t global_work_size[1] = {total_length};
	size_t local_work_size[1] = {256};
    for(size_t t_i=0; t_i < N; ++t_i) {
        // We don't go from beginning to end of array to simplify logic.

		if (t_i%2 == 0) {
			ret = clSetKernelArg(sim_kernel, 0, sizeof(cl_mem), &t1_buf);
		} else {
			ret = clSetKernelArg(sim_kernel, 0, sizeof(cl_mem), &t2_buf);
		}
		if (ret != CL_SUCCESS) {
			std::cerr << "Error setting kernel arg" << std::endl;
			std::cerr << opencl_errstr(ret) << std::endl;
			return 1;
		}
		if (t_i%2 == 0) {
			ret = clSetKernelArg(sim_kernel, 1, sizeof(cl_mem), &t2_buf);
		} else {
			ret = clSetKernelArg(sim_kernel, 1, sizeof(cl_mem), &t1_buf);
		}
		if (ret != CL_SUCCESS) {
			std::cerr << "Error setting kernel arg" << std::endl;
			std::cerr << opencl_errstr(ret) << std::endl;
			return 1;
		}

        ret = clEnqueueNDRangeKernel(command_queue, sim_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Error queuing kernel!" << std::endl;
			std::cerr << opencl_errstr(ret) << std::endl;
            return 1;
        }

        // Iterate time
        t += dt;
    }

    std::cout << "Simulation Finished." << std::endl;

    if (N%2 == 1) {
        ret = clEnqueueReadBuffer(command_queue, t2_buf, CL_TRUE, 0, total_length*sizeof(BUF_TYPE), t1, 0, NULL, NULL);
    } else {
        ret = clEnqueueReadBuffer(command_queue, t1_buf, CL_TRUE, 0, total_length*sizeof(BUF_TYPE), t1, 0, NULL, NULL);
    }
    if (ret != CL_SUCCESS) {
        std::cerr << "Error reading final buffer!" << std::endl;
		std::cerr << opencl_errstr(ret) << std::endl;
        return 1;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Read Buffer out" << std::endl;

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

    delete [] t1;

    std::cout << "Number of steps taken: " << N << std::endl;
    std::cout << "Final time: " << t << std::endl;
    std::cout << "Max value: " << buf_max << std::endl;
    std::cout << "Average value: " << avg << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    std::cout << "Took " << (duration.count()/((BUF_TYPE)N)) << " milliseconds per iteration" << std::endl;
    return 0;

}
