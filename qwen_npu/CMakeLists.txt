cmake_minimum_required(VERSION 3.16)
project(qwen_npu LANGUAGES CXX)
# 需要链接 CANN 的 glibc 动态库, 无法 musl 静态链接; so 通过 RPATH 自带依赖路径
set(CANN_HOME "$ENV{CANN_HOME}")
if(NOT CANN_HOME)
    message(FATAL_ERROR "❌ 需要设置环境变量 CANN_HOME")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${CANN_HOME}" CACHE STRING "CANN installation directory")
set(SOC_VERSION "Ascend310P3" CACHE STRING "Target Ascend SoC")
set(RUN_MODE "npu")
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include("${CANN_HOME}/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(qwen_npu_kernels STATIC kernels_npu.cpp)
ascendc_library(qwen_npu_kernels_tp2 STATIC kernels_tp2.cpp) # 双芯版 (QWEN_TP=2 由包装文件定义)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/qwen.cpp")
    add_library(qwen_npu SHARED qwen.cpp)
    add_library(qwen_npu2 SHARED qwen.cpp) # 双芯张量并行版
    target_compile_definitions(qwen_npu2 PRIVATE QWEN_TP=2)
    target_link_libraries(qwen_npu PRIVATE qwen_npu_kernels)
    target_link_libraries(qwen_npu2 PRIVATE qwen_npu_kernels_tp2)
    foreach(tgt qwen_npu qwen_npu2)
        target_compile_features(${tgt} PRIVATE cxx_std_17)
        target_include_directories(${tgt} PRIVATE "${CANN_HOME}/x86_64-linux/include")
        target_link_directories(${tgt} PRIVATE "${CANN_HOME}/x86_64-linux/lib64")
        target_link_libraries(${tgt} PRIVATE ascendcl pthread)
        target_link_options(${tgt} PRIVATE
            -Wl,-rpath,${CANN_HOME}/x86_64-linux/lib64
            -Wl,-rpath,/usr/local/Ascend/driver/lib64/driver
            -Wl,-rpath,/usr/local/Ascend/driver/lib64/common
            -Wl,-rpath-link,${CANN_HOME}/x86_64-linux/lib64
            -Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver
            -Wl,-rpath-link,/usr/local/Ascend/driver/lib64/common)
        set_target_properties(${tgt} PROPERTIES OUTPUT_NAME ${tgt} PREFIX "lib")
    endforeach()
endif()
