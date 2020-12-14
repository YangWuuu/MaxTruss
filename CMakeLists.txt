cmake_minimum_required(VERSION 3.5)
project(kmax-truss)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

include_directories(include)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(SRC_FILES src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/ktruss.cpp src/kcore.cpp src/preprocess.cpp src/tricount.cpp src/util.cpp)

add_executable(kmax_truss_serial ${SRC_FILES})
target_compile_definitions(kmax_truss_serial PRIVATE SERIAL=)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")

    add_executable(kmax_truss_omp ${SRC_FILES})
    target_compile_options(kmax_truss_omp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(kmax_truss_omp PRIVATE ${OpenMP_CXX_FLAGS})

    add_executable(kron_gen src/kron_gen.cpp src/log.cpp)
    target_compile_options(kron_gen PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(kron_gen PRIVATE ${OpenMP_CXX_FLAGS})
endif ()

find_package(CUDA)
if (CUDA_FOUND)
    message("CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50,code=sm_50")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")
    if ((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75")
    endif ()
    if (OPENMP_FOUND)
        # find_package(OPenMP) does not automatically add OpenMP flags to CUDA
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    endif (OPENMP_FOUND)

    set(CPU_SRC_FILES src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/util.cpp)
    set(GPU_SRC_FILES cuda/kcore.cu cuda/preprocess.cu cuda/tricount.cu cuda/ktruss.cu)

    cuda_add_executable(kmax_truss_cuda ${CPU_SRC_FILES} ${GPU_SRC_FILES})
    target_compile_options(kmax_truss_cuda PRIVATE ${OpenMP_CXX_FLAGS} -DCUDA)
    target_link_libraries(kmax_truss_cuda ${OpenMP_CXX_FLAGS})
    target_include_directories(kmax_truss_cuda PUBLIC include)

else ()
    message("CUDA NOT FOUND")
endif ()
