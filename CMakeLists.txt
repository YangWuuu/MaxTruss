cmake_minimum_required(VERSION 2.8)
project(kmax-truss)

set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(include)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(SRC_FILES src/main.cpp src/log.cpp src/read_file.cpp src/graph.cpp src/ktruss.cpp src/kcore.cpp)

add_executable(kmax_truss ${SRC_FILES})

add_executable(kmax_truss_serial ${SRC_FILES})
target_compile_definitions(kmax_truss_serial PRIVATE SERIAL=)

add_executable(kron_gen src/kron_gen.cpp src/log.cpp)

find_package(CUDA)
if(CUDA_FOUND)
    message("CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_INCLUDE_DIRS})
    #    set(CUDA_NVCC_FLAGS -O3;-std=c++11;-L/usr/local/cuda/lib64;-lcudart)
    set(CUDA_NVCC_FLAGS -O3;-L/usr/local/cuda/lib64;-lcudart)
    #    set(CUDA_NVCC_FLAGS -O0;-G;-g;-std=c++11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")
    if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75")
    endif()
    if(OPENMP_FOUND)
        # find_package(OPenMP) does not automatically add OpenMP flags to CUDA
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    endif(OPENMP_FOUND)

    cuda_add_executable(test_cuda src/main.cu src/log.cpp)

else()
    message("CUDA NOT FOUND")
endif()
