cmake_minimum_required(VERSION 3.15)
project(arm_simd_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Google Benchmark
find_package(benchmark REQUIRED)

# 检测 ARM 架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(IS_ARM 1)
    message(STATUS "ARM64 detected - enabling NEON intrinsics")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(IS_ARM 1)
    message(STATUS "ARM32 detected - enabling NEON intrinsics")
else()
    set(IS_ARM 0)
    message(WARNING "Non-ARM architecture detected - NEON code may not compile")
endif()

# 源文件
add_executable(simd_bench
    src/benchmark.cpp
)

# 链接 Google Benchmark
target_link_libraries(simd_bench
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
)

# 设置编译选项
if(IS_ARM)
    target_compile_options(simd_bench PRIVATE
        -march=native
    )
endif()

# 启用所有警告
target_compile_options(simd_bench PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# 包含目录
target_include_directories(simd_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
