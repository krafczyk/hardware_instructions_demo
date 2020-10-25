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

std::string add_arrays_kernel = R"CODE(__kernel void array_add(__global const float* A, __global const float* B, __global float* C) {
    // Get the index of the current element to be processed
    int i = get_global_id(0);

    // Do the operation
    C[i] = A[i] + B[i];
})CODE";

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

//std::string sim_kernel = R"CODE(__kernel void sim_kernel(__global const BUF_TYPE* in_buf, __global const BUF_TYPE* out_buf, const size_t i, const size_t side_length, const BUF_TYPE alpha, const BUF_TYPE dx) {
//    const size_t y_i = i/side_length;
//    const size_t x_i = i-(y_i*side_length);
//    if ((x_i > 0)&&(x_i < side_length-1)&&(y_i > 0)&&(y_i < side_length-1)) {
//        const BUF_TYPE up = in_buf[Idx(x_i, y_i-1, side_length)];
//        const BUF_TYPE down = in_buf[Idx(x_i, y_i+1, side_length)];
//        const BUF_TYPE left = in_buf[Idx(x_i-1, y_i, side_length)];
//        const BUF_TYPE right = in_buf[Idx(x_i+1, y_i, side_length)];
//        const BUF_TYPE center = in_buf[Idx(x_i, y_i, side_length)];
//        const BUF_TYPE lap = (up+down+left+right-4*center)/(dx*dx);
//        const BUF_TYPE tdiff = alpha*lap;
//        // Compute update
//        out_buf[i] = in_buf[i]+tdiff;
//    }
//})CODE";

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

    ArgParse::ArgParser Parser("Laplace simulation CPU");
    Parser.AddArgument("-size/--buf-size", "Target size of buffers in MB.", &target_buf_size);
    Parser.AddArgument("-gpu", "Indicate you'd like to use a GPU, otherwise it'll use a cpu.", &gpu);

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
    //size_t total_length = 0;

    std::string sim_kernel_program = produce_sim_kernel_code(side_length, alpha, dx);

    std::cout << "Program:" << std::endl;
    std::cout << sim_kernel_program << std::endl;

    cl::Device device_to_use;
    if (gpu) {
        auto gpu_devices = get_devices(CL_DEVICE_TYPE_GPU);
        if (gpu_devices.size() == 0) {
            std::cerr << "No GPUs available!" << std::endl;
            return 1;
        }
        if (gpu_devices.size() == 1) {
            device_to_use = gpu_devices[0];
        }
        const size_t hostname_len = 1024;
        char hostname[hostname_len];
        gethostname(hostname, hostname_len);
        bool set_device = false;
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
        if (!set_device) {
            std::cerr << "Couldn't set the GPU device!" << std::endl;
        }
    } else {
        auto cpu_devices = get_devices(CL_DEVICE_TYPE_CPU);
        if (cpu_devices.size() == 0) {
            std::cerr << "No suitable CPUs available" << std::endl;
        } else {
            device_to_use = cpu_devices[0];
        }
    }

    std::cout << "Device chosen" << std::endl;

    // Create OpenCL context
    std::vector<cl::Device> device_container;
    device_container.push_back(device_to_use);

    std::cout << "Create Context" << std::endl;
    cl::Context context(device_container);
    std::cout << "Context created" << std::endl;

    // Build program on our context
    cl::Program simProgram(context, sim_kernel_program);
    std::cout << "Start program build" << std::endl;
    cl_int ret = simProgram.build();
    std::cout << "Program Built" << std::endl;
    if(ret != CL_SUCCESS) {
        std::cerr << "Problem building OpenCL Program!" << std::endl;
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = simProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto& pair: buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    /*
    BUF_TYPE* t1 = nullptr;
    BUF_TYPE* t2 = nullptr;

    // Initializing buffers
    total_length = side_length*side_length;
    t1 = new BUF_TYPE[side_length*side_length];
    t2 = new BUF_TYPE[side_length*side_length];
    for (size_t i = 0; i < total_length; ++i) {
        t1[i] = 0.;
        t2[i] = 0.;
    }
   
    // Initial conditions May not be directly in the middle.
    t1[Idx(side_length/2,side_length/2, side_length)] = 100.;       

    std::cout << "Start simulation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    BUF_TYPE t = 0.;
    for(size_t t_i=0; t_i < N; ++t_i) {
        // We don't go from beginning to end of array to simplify logic.
        for(size_t i = 0; i < total_length; ++i) {
            SimKernel(t1, t2, i, side_length, alpha, dx);
        }

        // Swap buffers
        BUF_TYPE* ttemp = t1;
        t1 = t2;
        t2 = ttemp;

        // Iterate time
        t += dt;
    }

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

    delete [] t1;
    delete [] t2;

    std::cout << "Number of steps taken: " << N << std::endl;
    std::cout << "Final time: " << t << std::endl;
    std::cout << "Max value: " << buf_max << std::endl;
    std::cout << "Average value: " << avg << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    std::cout << "Took " << (duration.count()/((BUF_TYPE)N)) << " milliseconds per iteration" << std::endl;
    */
    return 0;
}
