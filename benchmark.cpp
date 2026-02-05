#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// ARM NEON 头文件
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define HAS_NEON 1
#else
    #define HAS_NEON 0
#endif

// 测试数据大小
constexpr size_t DATA_SIZE = 1024 * 1024;  // 1M 元素

// 测试数据容器
std::vector<float> g_data1;
std::vector<float> g_data2;
std::vector<float> g_result;

// 初始化测试数据
void InitializeData() {
    static bool initialized = false;
    if (!initialized) {
        g_data1.resize(DATA_SIZE);
        g_data2.resize(DATA_SIZE);
        g_result.resize(DATA_SIZE);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_data1[i] = dis(gen);
            g_data2[i] = dis(gen);
        }
        initialized = true;
    }
}

//=============================================================================
// 标量实现 (Scalar - 无 SIMD)
//=============================================================================

static void Scalar_Add(benchmark::State& state) {
    InitializeData();
    for (auto _ : state) {
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_result[i] = g_data1[i] + g_data2[i];
        }
        benchmark::DoNotOptimize(g_result.data());
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void Scalar_Subtract(benchmark::State& state) {
    InitializeData();
    for (auto _ : state) {
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_result[i] = g_data1[i] - g_data2[i];
        }
        benchmark::DoNotOptimize(g_result.data());
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void Scalar_Multiply(benchmark::State& state) {
    InitializeData();
    for (auto _ : state) {
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_result[i] = g_data1[i] * g_data2[i];
        }
        benchmark::DoNotOptimize(g_result.data());
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void Scalar_Divide(benchmark::State& state) {
    InitializeData();
    for (auto _ : state) {
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_result[i] = g_data1[i] / g_data2[i];
        }
        benchmark::DoNotOptimize(g_result.data());
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void Scalar_Sqrt(benchmark::State& state) {
    InitializeData();
    for (auto _ : state) {
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            g_result[i] = std::sqrt(std::abs(g_data1[i]));
        }
        benchmark::DoNotOptimize(g_result.data());
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

//=============================================================================
// ARM NEON SIMD 实现
//=============================================================================

#if HAS_NEON

static void NEON_Add(benchmark::State& state) {
    InitializeData();
    const float* src1 = g_data1.data();
    const float* src2 = g_data2.data();
    float* dst = g_result.data();

    for (auto _ : state) {
        size_t i = 0;
        // 处理 16 个元素为一个块 (4x float32x4_t)
        for (; i + 15 < DATA_SIZE; i += 16) {
            float32x4_t a1 = vld1q_f32(src1 + i);
            float32x4_t a2 = vld1q_f32(src1 + i + 4);
            float32x4_t a3 = vld1q_f32(src1 + i + 8);
            float32x4_t a4 = vld1q_f32(src1 + i + 12);

            float32x4_t b1 = vld1q_f32(src2 + i);
            float32x4_t b2 = vld1q_f32(src2 + i + 4);
            float32x4_t b3 = vld1q_f32(src2 + i + 8);
            float32x4_t b4 = vld1q_f32(src2 + i + 12);

            float32x4_t r1 = vaddq_f32(a1, b1);
            float32x4_t r2 = vaddq_f32(a2, b2);
            float32x4_t r3 = vaddq_f32(a3, b3);
            float32x4_t r4 = vaddq_f32(a4, b4);

            vst1q_f32(dst + i, r1);
            vst1q_f32(dst + i + 4, r2);
            vst1q_f32(dst + i + 8, r3);
            vst1q_f32(dst + i + 12, r4);
        }
        // 处理剩余元素
        for (; i < DATA_SIZE; ++i) {
            dst[i] = src1[i] + src2[i];
        }
        benchmark::DoNotOptimize(dst);
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void NEON_Subtract(benchmark::State& state) {
    InitializeData();
    const float* src1 = g_data1.data();
    const float* src2 = g_data2.data();
    float* dst = g_result.data();

    for (auto _ : state) {
        size_t i = 0;
        for (; i + 15 < DATA_SIZE; i += 16) {
            float32x4_t a1 = vld1q_f32(src1 + i);
            float32x4_t a2 = vld1q_f32(src1 + i + 4);
            float32x4_t a3 = vld1q_f32(src1 + i + 8);
            float32x4_t a4 = vld1q_f32(src1 + i + 12);

            float32x4_t b1 = vld1q_f32(src2 + i);
            float32x4_t b2 = vld1q_f32(src2 + i + 4);
            float32x4_t b3 = vld1q_f32(src2 + i + 8);
            float32x4_t b4 = vld1q_f32(src2 + i + 12);

            float32x4_t r1 = vsubq_f32(a1, b1);
            float32x4_t r2 = vsubq_f32(a2, b2);
            float32x4_t r3 = vsubq_f32(a3, b3);
            float32x4_t r4 = vsubq_f32(a4, b4);

            vst1q_f32(dst + i, r1);
            vst1q_f32(dst + i + 4, r2);
            vst1q_f32(dst + i + 8, r3);
            vst1q_f32(dst + i + 12, r4);
        }
        for (; i < DATA_SIZE; ++i) {
            dst[i] = src1[i] - src2[i];
        }
        benchmark::DoNotOptimize(dst);
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void NEON_Multiply(benchmark::State& state) {
    InitializeData();
    const float* src1 = g_data1.data();
    const float* src2 = g_data2.data();
    float* dst = g_result.data();

    for (auto _ : state) {
        size_t i = 0;
        for (; i + 15 < DATA_SIZE; i += 16) {
            float32x4_t a1 = vld1q_f32(src1 + i);
            float32x4_t a2 = vld1q_f32(src1 + i + 4);
            float32x4_t a3 = vld1q_f32(src1 + i + 8);
            float32x4_t a4 = vld1q_f32(src1 + i + 12);

            float32x4_t b1 = vld1q_f32(src2 + i);
            float32x4_t b2 = vld1q_f32(src2 + i + 4);
            float32x4_t b3 = vld1q_f32(src2 + i + 8);
            float32x4_t b4 = vld1q_f32(src2 + i + 12);

            float32x4_t r1 = vmulq_f32(a1, b1);
            float32x4_t r2 = vmulq_f32(a2, b2);
            float32x4_t r3 = vmulq_f32(a3, b3);
            float32x4_t r4 = vmulq_f32(a4, b4);

            vst1q_f32(dst + i, r1);
            vst1q_f32(dst + i + 4, r2);
            vst1q_f32(dst + i + 8, r3);
            vst1q_f32(dst + i + 12, r4);
        }
        for (; i < DATA_SIZE; ++i) {
            dst[i] = src1[i] * src2[i];
        }
        benchmark::DoNotOptimize(dst);
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void NEON_Divide(benchmark::State& state) {
    InitializeData();
    const float* src1 = g_data1.data();
    const float* src2 = g_data2.data();
    float* dst = g_result.data();

    for (auto _ : state) {
        size_t i = 0;
        // NEON 没有直接的除法指令，使用倒数近似
        for (; i + 15 < DATA_SIZE; i += 16) {
            float32x4_t a1 = vld1q_f32(src1 + i);
            float32x4_t a2 = vld1q_f32(src1 + i + 4);
            float32x4_t a3 = vld1q_f32(src1 + i + 8);
            float32x4_t a4 = vld1q_f32(src1 + i + 12);

            float32x4_t b1 = vld1q_f32(src2 + i);
            float32x4_t b2 = vld1q_f32(src2 + i + 4);
            float32x4_t b3 = vld1q_f32(src2 + i + 8);
            float32x4_t b4 = vld1q_f32(src2 + i + 12);

            // 使用近似除法
            float32x4_t r1 = vdivq_f32(a1, b1);
            float32x4_t r2 = vdivq_f32(a2, b2);
            float32x4_t r3 = vdivq_f32(a3, b3);
            float32x4_t r4 = vdivq_f32(a4, b4);

            vst1q_f32(dst + i, r1);
            vst1q_f32(dst + i + 4, r2);
            vst1q_f32(dst + i + 8, r3);
            vst1q_f32(dst + i + 12, r4);
        }
        for (; i < DATA_SIZE; ++i) {
            dst[i] = src1[i] / src2[i];
        }
        benchmark::DoNotOptimize(dst);
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

static void NEON_Sqrt(benchmark::State& state) {
    InitializeData();
    const float* src1 = g_data1.data();
    float* dst = g_result.data();

    for (auto _ : state) {
        size_t i = 0;
        for (; i + 15 < DATA_SIZE; i += 16) {
            float32x4_t a1 = vld1q_f32(src1 + i);
            float32x4_t a2 = vld1q_f32(src1 + i + 4);
            float32x4_t a3 = vld1q_f32(src1 + i + 8);
            float32x4_t a4 = vld1q_f32(src1 + i + 12);

            // 先取绝对值，再开方
            float32x4_t abs1 = vabsq_f32(a1);
            float32x4_t abs2 = vabsq_f32(a2);
            float32x4_t abs3 = vabsq_f32(a3);
            float32x4_t abs4 = vabsq_f32(a4);

            float32x4_t r1 = vsqrtq_f32(abs1);
            float32x4_t r2 = vsqrtq_f32(abs2);
            float32x4_t r3 = vsqrtq_f32(abs3);
            float32x4_t r4 = vsqrtq_f32(abs4);

            vst1q_f32(dst + i, r1);
            vst1q_f32(dst + i + 4, r2);
            vst1q_f32(dst + i + 8, r3);
            vst1q_f32(dst + i + 12, r4);
        }
        for (; i < DATA_SIZE; ++i) {
            dst[i] = std::sqrt(std::abs(src1[i]));
        }
        benchmark::DoNotOptimize(dst);
    }
    state.SetItemsProcessed(DATA_SIZE * state.iterations());
}

#endif // HAS_NEON

//=============================================================================
// 注册 Benchmark
//=============================================================================

// 标量版本
BENCHMARK(Scalar_Add)->Unit(benchmark::kMicrosecond);
BENCHMARK(Scalar_Subtract)->Unit(benchmark::kMicrosecond);
BENCHMARK(Scalar_Multiply)->Unit(benchmark::kMicrosecond);
BENCHMARK(Scalar_Divide)->Unit(benchmark::kMicrosecond);
BENCHMARK(Scalar_Sqrt)->Unit(benchmark::kMicrosecond);

// NEON SIMD 版本
#if HAS_NEON
BENCHMARK(NEON_Add)->Unit(benchmark::kMicrosecond);
BENCHMARK(NEON_Subtract)->Unit(benchmark::kMicrosecond);
BENCHMARK(NEON_Multiply)->Unit(benchmark::kMicrosecond);
BENCHMARK(NEON_Divide)->Unit(benchmark::kMicrosecond);
BENCHMARK(NEON_Sqrt)->Unit(benchmark::kMicrosecond);
#endif

BENCHMARK_MAIN();
