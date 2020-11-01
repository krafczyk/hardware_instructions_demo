cmake_minimum_required(VERSION 3.10)

project(hardware_instructions_demo VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenMP REQUIRED)
find_package(OpenCL)
find_package(MKL)
set(BLA_VENDOR "OpenBLAS")
find_package(BLAS)

# Buffer type list
set(buf_kinds "0" "1")
set(buf_types "float" "double")

# Set of base compile options
#set(base_compile_options1 "-Wall" "-Werror" "-Wextra" "-fverbose-asm" "-fopt-info")
set(base_compile_options1 "-Wall" "-Werror" "-Wextra" "-Wno-ignored-attributes" "-fverbose-asm")
set(base_compile_options1_cuda "-Wall" "-Wno-ignored-attributes" "-fverbose-asm")
set(omp_compile_options "-fopenmp")
set(omp_link_options "-lgomp -lpthread")
set(acc_link_options "-lpthread")

# program group 1
set(compile_versions1 O3_i386 O3_i686 O3_nehalem O3_broadwell O3_skylake O3_knl)
set(compile_options1 "-O3 -march=i386 -m32" "-O3 -march=i686 -m32" "-O3 -march=nehalem" "-O3 -march=broadwell" "-O3 -march=skylake" "-O3 -march=knl")
set(link_options1 "-m32" "-m32" "_" "_" "_" "_")

# Compute lengths of arrays
list(LENGTH compile_versions1 versions1_length)
math(EXPR versions1_len "${versions1_length} - 1")
list(LENGTH compile_versions2 versions2_length)
math(EXPR versions2_len "${versions2_length} - 1")
list(LENGTH buf_kinds buf_kinds_length)
math(EXPR buf_kinds_len "${buf_kinds_length} - 1")

# Loop over all buffer types
foreach(buf_idx RANGE ${buf_kinds_len})
list(GET buf_kinds ${buf_idx} buf_kind)
list(GET buf_types ${buf_idx} buf_type)

# Handle laplace cuda on different types
add_executable(laplace_cuda_${buf_type} laplace_cuda.cu)
target_compile_options(laplace_cuda_${buf_type} PRIVATE ${base_compile_options1_cuda})
target_compile_options(laplace_cuda_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
set_property(TARGET laplace_cuda_${buf_type} PROPERTY CUDA_STANDARD 11)
set_property(TARGET laplace_cuda_${buf_type} PROPERTY CUDA_ARCHITECTURES 61)

# Loop over the different compile options
foreach(idx RANGE ${versions1_len})
list(GET compile_versions1 ${idx} compile_version)
list(GET compile_options1 ${idx} compile_options)
list(GET link_options1 ${idx} link_options)
separate_arguments(compile_options)

# loop1
add_executable(loop1_${compile_version}_${buf_type} loop1.cpp)
target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_${compile_version}_${buf_type} PRIVATE ${compile_options})
if("${link_options}" MATCHES "^.*-m32.*$")
target_link_libraries(loop1_${compile_version}_${buf_type} PRIVATE ${link_options})
endif()

# loop1 mkl
if(MKL_FOUND)
if(NOT "${link_options}" MATCHES "^.*-m32.*$")
add_executable(loop1_mkl_${compile_version}_${buf_type} loop1_mkl.cpp)
target_compile_options(loop1_mkl_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop1_mkl_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_mkl_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(loop1_mkl_${compile_version}_${buf_type} PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(loop1_mkl_${compile_version}_${buf_type} PRIVATE "${link_options}")
target_link_libraries(loop1_mkl_${compile_version}_${buf_type} PRIVATE "${MKL_LIBRARIES}")
endif()
endif()

# loop1 mkl2
if(MKL_FOUND)
if(NOT "${link_options}" MATCHES "^.*-m32.*$")
add_executable(loop1_mkl2_${compile_version}_${buf_type} loop1_mkl2.cpp)
target_compile_options(loop1_mkl2_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop1_mkl2_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_mkl2_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(loop1_mkl2_${compile_version}_${buf_type} PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(loop1_mkl2_${compile_version}_${buf_type} PRIVATE "${MKL_LIBRARIES}")
endif()
endif()

# loop1 openblas
if(BLAS_FOUND)
if(NOT "${link_options}" MATCHES "^.*-m32.*$")
add_executable(loop1_openblas_${compile_version}_${buf_type} loop1_openblas.cpp)
target_compile_options(loop1_openblas_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop1_openblas_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_openblas_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(loop1_openblas_${compile_version}_${buf_type} PRIVATE ${BLAS_INCLUDE_DIR})
target_link_libraries(loop1_openblas_${compile_version}_${buf_type} PRIVATE "${BLAS_LIBRARIES}")
endif()
endif()

# loop1 opencl
if(OpenCL_FOUND)
if(NOT "${link_options}" MATCHES "^.*-m32.*$")
add_executable(loop1_opencl_${compile_version}_${buf_type} loop1_opencl.cpp)
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_include_directories(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(loop1_opencl_${compile_version}_${buf_type} PRIVATE ${OpenCL_LIBRARIES})
endif()
endif()

# loop2
add_executable(loop2_${compile_version}_${buf_type} loop2.cpp)
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(loop2_${compile_version}_${buf_type} PRIVATE ${compile_options})
if("${link_options}" MATCHES "^.*-m32.*$")
target_link_libraries(loop2_${compile_version}_${buf_type} PRIVATE ${link_options})
endif()

# laplace
add_executable(laplace_${compile_version}_${buf_type} laplace.cpp)
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_compile_options(laplace_${compile_version}_${buf_type} PRIVATE ${omp_compile_options})
target_link_libraries(laplace_${compile_version}_${buf_type} PRIVATE "${omp_link_options}")
if("${link_options}" MATCHES "^.*-m32.*$")
target_link_libraries(laplace_${compile_version}_${buf_type} PRIVATE "${link_options}")
endif()

# laplace OpenCL
if(OpenCL_FOUND)
if(NOT "${link_options}" MATCHES "^.*-m32.*$")
add_executable(laplace_opencl_${compile_version}_${buf_type} laplace_opencl.cpp)
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_include_directories(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${compile_options})
target_link_libraries(laplace_opencl_${compile_version}_${buf_type} PRIVATE ${OpenCL_LIBRARIES})
endif()
endif()

# laplace OpenACC
add_executable(laplace_openacc_${compile_version}_${buf_type} laplace_openacc.cpp)
target_compile_options(laplace_openacc_${compile_version}_${buf_type} PRIVATE "-DBUF_KIND=${buf_kind}")
target_compile_options(laplace_openacc_${compile_version}_${buf_type} PRIVATE "-fopenacc")
target_compile_options(laplace_openacc_${compile_version}_${buf_type} PRIVATE ${base_compile_options1})
target_compile_options(laplace_openacc_${compile_version}_${buf_type} PRIVATE ${compile_options})
if("${link_options}" MATCHES "^.*-m32.*$")
target_link_libraries(laplace_openacc_${compile_version}_${buf_type} PRIVATE "${link_options}")
endif()
target_link_libraries(laplace_openacc_${compile_version}_${buf_type} PRIVATE "${omp_link_options}")

endforeach()
endforeach()

add_executable(loop1_cuda loop1_cuda.cu)
set_property(TARGET loop1_cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET loop1_cuda PROPERTY CUDA_ARCHITECTURES 61)
