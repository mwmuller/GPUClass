cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

project(TiledMatrixMultiplication)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

find_package(CUDA REQUIRED)

if (NOT WIN32)
    find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.0 Maxwell 6.1)

#add_definitions(-DWB_USE_CUDA)

find_path(LIBWB_ROOT NAMES wb.h HINTS libwb ../libwb ../../libwb)
include_directories(${LIBWB_ROOT})
find_path(LIBWB_LIB_DIR NAMES libwb.so HINTS ${LIBWB_ROOT}/lib ${LIBWB_ROOT}/build) 
link_directories(${LIBWB_LIB_DIR})

set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "--use_fast_math")

cuda_add_executable(TiledMatrixMultiplication TiledMatrixMultiplication.cu)
target_link_libraries(TiledMatrixMultiplication wb rt)
