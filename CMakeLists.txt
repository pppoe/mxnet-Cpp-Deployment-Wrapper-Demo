cmake_minimum_required (VERSION 2.8)

project (MXNetCppDemo)

add_definitions (-Wno-deprecated-register)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")

set (USE_MKL False) # use openblas by default

## Change this for your own settings - you can use brew install openblas, brew install opencv3 for these on Mac
set (OpenBlasRoot /usr/local/Cellar/openblas/0.2.15/)
set (OpenCVRoot /usr/local/Cellar/opencv3/3.1.0_1/)

if (${USE_MKL})
    message("...using MKL in $ENV{MKLROOT}")
    include_directories ($ENV{MKLROOT}/include)
    set (MKL_LIB_PATH $ENV{MKLROOT}/lib/intel64)
    link_directories (${MKL_LIB_PATH})
    set (STATIC_LIBS 
        -Wl,--start-group
        ${MKL_LIB_PATH}/libmkl_intel_lp64.a 
        ${MKL_LIB_PATH}/libmkl_core.a 
        ${MKL_LIB_PATH}/libmkl_intel_thread.a 
        -Wl,--end-group
        -liomp5 
        -ldl -lpthread)
else()
    message("...using OpenBlas in ${OpenBlasRoot}")
    include_directories (${OpenBlasRoot}/include)
    link_directories (${OpenBlasRoot}/lib/)
    set (LD_FLAGS ${LD_FLAGS} openblas)
endif()

message("...using OpenCV in ${OpenCVRoot}")
include_directories(${OpenCVRoot}/include)
link_directories(${OpenCVRoot}/lib)
set (OpenCV_LIBS opencv_core;opencv_imgproc;opencv_imgcodecs)
set (LD_FLAGS ${LD_FLAGS} ${OpenCV_LIBS})

### MXNET
include_directories (MXNet-v0.11.0/)
add_library(mxnet MXNet-v0.11.0/mxnet_predict-all.cc)
if (${USE_MKL})
    target_link_libraries (mxnet ${STATIC_LIBS})
else()
    target_link_libraries (mxnet ${LD_FLAGS})
endif()
set (LD_FLAGS mxnet ${LD_FLAGS})
    
add_executable (main main.cpp mx_model.cc)
target_link_libraries (main ${LD_FLAGS} ${STATIC_LIBS})
