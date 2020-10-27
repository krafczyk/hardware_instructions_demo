cmake_minimum_required(VERSION 3.10)

project(hardware_instructions_demo VERSION 1.0 LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Buffer type list
set(buf_types "float" "double")

# Set of base compile options
set(base_compile_options1 "-std=c++11" "-Wall" "-Werror" "-Wextra" "-fverbose-asm")
set(omp_compile_options "-fopenmp")
set(omp_link_options "-lgomp -lpthread")

# program group 1
set(compile_versions1 O3_i386 O3_i686 O3_nehalem O3_broadwell O3_skylake O3_knl)
set(compile_options1 "-O3 -march=i386 -m32" "-O3 -march=i686 -m32" "-O3 -march=nehalem" "-O3 -march=broadwell" "-O3 -march=skylake" "-O3 -march=knl")
set(link_options1 "-m32" "-m32" "_" "_" "_" "_")

# Compute lengths of arrays
list(LENGTH compile_versions1 versions1_length)
math(EXPR versions1_len "${versions1_length} - 1")
list(LENGTH compile_versions2 versions2_length)
math(EXPR versions2_len "${versions2_length} - 1")
list(LENGTH buf_types buf_types_length)
math(EXPR buf_types_len "${buf_types_length} - 1")

foreach(buf_idx RANGE ${buf_types_len})
list(GET buf_types ${buf_idx} buf_type)

add_executable(laplace_cuda_${buf_type} laplace_cuda.cu)
target_compile_options(laplace_cuda_${buf_type} PRIVATE ${base_compile_option1})
target_compile_options(laplace_cuda_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")
set_property(TARGET laplace_cuda_${buf_type} PROPERTY CUDA_ARCHITECTURES 61)

foreach(idx RANGE ${versions1_len})
list(GET compile_versions1 ${idx} compile_version)
list(GET compile_options1 ${idx} compile_options)
list(GET link_options1 ${idx} link_options)
separate_arguments(compile_options)

add_executable(loop1_${compile_version}_${buf_type} loop1.cpp)
add_executable(loop2_${compile_version}_${buf_type} loop2.cpp)
add_executable(loop1_opencl_${compile_version}_${buf_type} loop1_opencl.cpp)
add_executable(laplace_${compile_version}_${buf_type} laplace.cpp)
add_executable(laplace_opencl_${compile_version}_${buf_type} laplace_opencl.cpp)

target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE "-DBUF_TYPE=${buf_type}")

target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})

target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${omp_compile_options})

target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${compile_options})
if(NOT "${link_options}" STREQUAL "_")
set_target_properties(loop1_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS ${link_options})
set_target_properties(loop2_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS ${link_options})
set_target_properties(loop1_opencl_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "${link_options} -lOpenCL")
set_target_properties(laplace_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "${link_options} ${omp_link_options}")
set_target_properties(laplace_opencl_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "${link_options} -lOpenCL")
else()
set_target_properties(loop1_opencl_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "-lOpenCL")
set_target_properties(laplace_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "${omp_link_options}")
set_target_properties(laplace_opencl_${compile_version}_${buf_type} PROPERTIES LINK_FLAGS "-lOpenCL")
endif()
endforeach()
endforeach()

add_executable(loop1_cuda loop1_cuda.cu)
set_property(TARGET loop1_cuda PROPERTY CUDA_ARCHITECTURES 61)
