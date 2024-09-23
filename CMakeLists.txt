cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)
project(EKF)

set(CUDAToolkit_ROOT /usr/local/cuda)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_SOURCES
    src/ekf.cu
)
set(SOURCES
    src/ekf.cpp
)

add_library(ekf ${SOURCES} ${CUDA_SOURCES})

target_include_directories(ekf PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ekf PRIVATE ${CUDA_LIBRARIES} nvinfer)

