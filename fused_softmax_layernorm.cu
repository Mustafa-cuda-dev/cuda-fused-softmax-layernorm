
#include <cuda_runtime.h>      
#include <iostream>      
#include <cmath>      
#include <vector>      
#include <iomanip>      
#include <algorithm>      
#include <cstdlib>      
      
#define CUDA_CHECK(call) \      
    do { \      
        cudaError_t err = call; \      
        if (err != cudaSuccess) { \      
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \      
                      << " - " << cudaGetErrorString(err) << std::endl; \      
            exit(EXIT_FAILURE); \      
        } \      
    } while (0)      
      
#define CUDA_POST_KERNEL_CHECK() \      
    do { \      
        cudaError_t err = cudaGetLastError(); \      
        if (err != cudaSuccess) { \      
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":" << __LINE__ \      
                      << " - " << cudaGetErrorString(err) << std::endl; \      
            exit(EXIT_FAILURE); \      
        } \      
    } while (0)      
      
// ============================================================================      
// REDUCTION UTILITIES      
// ============================================================================      
      
__device__ inline float blockReduceMax(float val, float* s_scratch) {      
    int tid = threadIdx.x;      
    int lane = tid & 31;      
    int wid = tid >> 5;      
    int num_warps = (blockDim.x + 31) >> 5;      
    int num_threads_in_warp = min(32, (int)blockDim.x - (wid << 5));      
      
    unsigned int warp_mask =      
        (blockDim.x - (wid << 5) >= 32)      
            ? 0xffffffffu      
            : (1U << (blockDim.x & 31)) - 1U;      
      
    for (int offset = 16; offset > 0; offset >>= 1) {      
        float shfl_val = __shfl_down_sync(warp_mask, val, offset);      
        if (lane + offset < num_threads_in_warp) {      
            val = fmaxf(val, shfl_val);      
        }      
    }      
      
    if (lane == 0) {      
        s_scratch[wid] = val;      
    }      
    __syncthreads();      
      
    if (wid == 0) {      
        float v = (lane < num_warps) ? s_scratch[lane] : -INFINITY;      
      
        // Corrected: shift count is always bounded below 32.      
        unsigned int warp0_mask =      
            (blockDim.x >= 32)      
                ? 0xffffffffu      
                : (1U << (blockDim.x & 31)) - 1U;      
      
        int warp0_threads = min(32, (int)blockDim.x);      
      
        for (int offset = 16; offset > 0; offset >>= 1) {      
            float shfl_val = __shfl_down_sync(warp0_mask, v, offset);      
            if (lane + offset < warp0_threads) {      
                v = fmaxf(v, shfl_val);      
            }      
        }      
      
        if (lane == 0) {      
            s_scratch[0] = v;      
        }      
    }      
      
    __syncthreads();      
      
    // Cross-call WAR protection.      
    float result = s_scratch[0];      
    __syncthreads();      
      
    return result;      
}      
      
__device__ inline float blockReduceSum(float val, float* s_scratch) {      
    int tid = threadIdx.x;      
    int lane = tid & 31;      
    int wid = tid >> 5;      
    int num_warps = (blockDim.x + 31) >> 5;      
    int num_threads_in_warp = min(32, (int)blockDim.x - (wid << 5));      
      
    unsigned int warp_mask =      
        (blockDim.x - (wid << 5) >= 32)      
            ? 0xffffffffu      
            : (1U << (blockDim.x & 31)) - 1U;      
      
    for (int offset = 16; offset > 0; offset >>= 1) {      
        float shfl_val = __shfl_down_sync(warp_mask, val, offset);      
        if (lane + offset < num_threads_in_warp) {      
            val += shfl_val;      
        }      
    }      
      
    if (lane == 0) {      
        s_scratch[wid] = val;      
    }      
    __syncthreads();      
      
    if (wid == 0) {      
        float v = (lane < num_warps) ? s_scratch[lane] : 0.0f;      
      
        // Corrected: shift count is always bounded below 32.      
        unsigned int warp0_mask =      
            (blockDim.x >= 32)      
                ? 0xffffffffu      
                : (1U << (blockDim.x & 31)) - 1U;      
      
        int warp0_threads = min(32, (int)blockDim.x);      
      
        for (int offset = 16; offset > 0; offset >>= 1) {      
            float shfl_val = __shfl_down_sync(warp0_mask, v, offset);      
            if (lane + offset < warp0_threads) {      
                v += shfl_val;      
            }      
        }      
      
        if (lane == 0) {      
            s_scratch[0] = v;      
        }      
    }      
      
    __syncthreads();      
      
    // Cross-call WAR protection.      
    float result = s_scratch[0];      
    __syncthreads();      
      
    return result;      
}      
      
// ============================================================================      
// UNFUSED BASELINE      
// ============================================================================      
      
__global__ void __launch_bounds__(256, 4)      
softmax_kernel(      
    const float* __restrict__ X,      
    float* __restrict__ Y,      
    int N,      
    int D      
) {      
    int row = blockIdx.x;      
    if (row >= N) return;      
      
    extern __shared__ float s_scratch[];      
      
    int tid = threadIdx.x;      
      
    const float* row_in = X + (size_t)row * D;      
    float* row_out = Y + (size_t)row * D;      
      
    float local_max = -INFINITY;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        local_max = fmaxf(local_max, row_in[j]);      
    }      
      
    float row_max = blockReduceMax(local_max, s_scratch);      
      
    float local_sum = 0.0f;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        local_sum += __expf(row_in[j] - row_max);      
    }      
      
    float sum_exp = blockReduceSum(local_sum, s_scratch);      
    float inv_sum_exp = 1.0f / sum_exp;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        row_out[j] =      
            __expf(row_in[j] - row_max) * inv_sum_exp;      
    }      
}      
      
__global__ void __launch_bounds__(256, 4)      
layernorm_kernel(      
    const float* __restrict__ Y,      
    float* __restrict__ Z,      
    const float* __restrict__ gamma,      
    const float* __restrict__ beta,      
    int N,      
    int D,      
    float eps      
) {      
    int row = blockIdx.x;      
    if (row >= N) return;      
      
    extern __shared__ float s_scratch[];      
      
    int tid = threadIdx.x;      
      
    const float* row_in = Y + (size_t)row * D;      
    float* row_out = Z + (size_t)row * D;      
      
    float local_sum = 0.0f;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        local_sum += row_in[j];      
    }      
      
    float mean =      
        blockReduceSum(local_sum, s_scratch) / D;      
      
    float local_var_sum = 0.0f;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        float diff = row_in[j] - mean;      
        local_var_sum += diff * diff;      
    }      
      
    float variance =      
        blockReduceSum(local_var_sum, s_scratch) / D;      
      
    float rstd = rsqrtf(variance + eps);      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        row_out[j] =      
            (row_in[j] - mean) *      
            rstd *      
            gamma[j] +      
            beta[j];      
    }      
}      
      
// ============================================================================      
// FUSED SOFTMAX + LAYERNORM      
// ============================================================================      
      
__global__ void __launch_bounds__(256, 4)      
fused_softmax_layernorm_kernel(      
    const float* __restrict__ X,      
    float* __restrict__ Z,      
    const float* __restrict__ gamma,      
    const float* __restrict__ beta,      
    int N,      
    int D,      
    float eps      
) {      
    int row = blockIdx.x;      
    if (row >= N) return;      
      
    extern __shared__ char shared_pool[];      
      
    float* s_scratch =      
        reinterpret_cast<float*>(shared_pool);      
      
    float* s_y =      
        reinterpret_cast<float*>(      
            shared_pool + 32 * sizeof(float)      
        );      
      
    int tid = threadIdx.x;      
      
    const float* row_in =      
        X + (size_t)row * D;      
      
    float* row_out =      
        Z + (size_t)row * D;      
      
    float local_max = -INFINITY;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        local_max =      
            fmaxf(local_max, row_in[j]);      
    }      
      
    float row_max =      
        blockReduceMax(local_max, s_scratch);      
      
    float local_sum = 0.0f;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        float val =      
            __expf(row_in[j] - row_max);      
      
        s_y[j] = val;      
        local_sum += val;      
    }      
      
    float sum_exp =      
        blockReduceSum(local_sum, s_scratch);      
      
    float inv_sum_exp =      
        1.0f / sum_exp;      
      
    float mean =      
        1.0f / static_cast<float>(D);      
      
    float local_var_sum = 0.0f;      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        float y_val =      
            s_y[j] * inv_sum_exp;      
      
        s_y[j] = y_val;      
      
        float diff =      
            y_val - mean;      
      
        local_var_sum +=      
            diff * diff;      
    }      
      
    float variance =      
        blockReduceSum(      
            local_var_sum,      
            s_scratch      
        ) / D;      
      
    float rstd =      
        rsqrtf(variance + eps);      
      
    for (int j = tid; j < D; j += blockDim.x) {      
        row_out[j] =      
            (s_y[j] - mean) *      
            rstd *      
            gamma[j] +      
            beta[j];      
    }      
}      
      
// ============================================================================      
// BENCHMARK AND VALIDATION      
// ============================================================================      
      
void run_benchmark(int N, int D) {      
      
    std::cout      
        << "\n========================================\n";      
      
    std::cout      
        << "Benchmarking Config: N = "      
        << N      
        << ", D = "      
        << D      
        << "\n";      
      
    std::cout      
        << "========================================\n";      
      
    size_t num_elements =      
        (size_t)N * (size_t)D;      
      
    size_t size_X =      
        num_elements * sizeof(float);      
      
    size_t size_param =      
        (size_t)D * sizeof(float);      
      
    std::vector<float> h_X(num_elements);      
      
    std::vector<float>      
        h_gamma((size_t)D, 1.0f);      
      
    std::vector<float>      
        h_beta((size_t)D, 0.0f);      
      
    std::vector<float>      
        h_out_unfused(num_elements, 0.0f);      
      
    std::vector<float>      
        h_out_fused(num_elements, 0.0f);      
      
    for (size_t i = 0; i < num_elements; ++i) {      
        h_X[i] =      
            static_cast<float>(rand()) /      
            static_cast<float>(RAND_MAX) *      
            2.0f -      
            1.0f;      
    }      
      
    float* d_X = nullptr;      
    float* d_Y_temp = nullptr;      
    float* d_Z_unfused = nullptr;      
    float* d_Z_fused = nullptr;      
    float* d_gamma = nullptr;      
    float* d_beta = nullptr;      
      
    CUDA_CHECK(cudaMalloc(&d_X, size_X));      
    CUDA_CHECK(cudaMalloc(&d_Y_temp, size_X));      
    CUDA_CHECK(cudaMalloc(&d_Z_unfused, size_X));      
    CUDA_CHECK(cudaMalloc(&d_Z_fused, size_X));      
    CUDA_CHECK(cudaMalloc(&d_gamma, size_param));      
    CUDA_CHECK(cudaMalloc(&d_beta, size_param));      
      
    CUDA_CHECK(      
        cudaMemcpy(      
            d_X,      
            h_X.data(),      
            size_X,      
            cudaMemcpyHostToDevice      
        )      
    );      
      
    CUDA_CHECK(      
        cudaMemcpy(      
            d_gamma,      
            h_gamma.data(),      
            size_param,      
            cudaMemcpyHostToDevice      
        )      
    );      
      
    CUDA_CHECK(      
        cudaMemcpy(      
            d_beta,      
            h_beta.data(),      
            size_param,      
            cudaMemcpyHostToDevice      
        )      
    );      
      
    const int threads = 256;      
    const int blocks = N;      
      
    size_t scratch_shmem_size =      
        32 * sizeof(float);      
      
    size_t fused_shmem_size =      
        32 * sizeof(float) +      
        (size_t)D * sizeof(float);      
      
    int max_shared_mem = 0;      
    int device_id = 0;      
      
    CUDA_CHECK(cudaGetDevice(&device_id));      
      
    CUDA_CHECK(      
        cudaDeviceGetAttribute(      
            &max_shared_mem,      
            cudaDevAttrMaxSharedMemoryPerBlockOptin,      
            device_id      
        )      
    );      
      
    if (fused_shmem_size >      
        (size_t)max_shared_mem) {      
      
        std::cerr      
            << "Error: Requested shared memory size "      
            << fused_shmem_size      
            << " exceeds device maximum "      
            << max_shared_mem      
            << std::endl;      
      
        exit(EXIT_FAILURE);      
    }      
      
    CUDA_CHECK(      
        cudaFuncSetAttribute(      
            fused_softmax_layernorm_kernel,      
            cudaFuncAttributeMaxDynamicSharedMemorySize,      
            (int)fused_shmem_size      
        )      
    );      
      
    cudaEvent_t start;      
    cudaEvent_t stop;      
      
    CUDA_CHECK(cudaEventCreate(&start));      
    CUDA_CHECK(cudaEventCreate(&stop));      
      
    // Warmup      
    for (int i = 0; i < 10; ++i) {      
      
        softmax_kernel      
            <<<blocks, threads, scratch_shmem_size>>>(      
                d_X,      
                d_Y_temp,      
                N,      
                D      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
      
        layernorm_kernel      
            <<<blocks, threads, scratch_shmem_size>>>(      
                d_Y_temp,      
                d_Z_unfused,      
                d_gamma,      
                d_beta,      
                N,      
                D,      
                1e-5f      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
      
        fused_softmax_layernorm_kernel      
            <<<blocks, threads, fused_shmem_size>>>(      
                d_X,      
                d_Z_fused,      
                d_gamma,      
                d_beta,      
                N,      
                D,      
                1e-5f      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
    }      
      
    CUDA_CHECK(cudaDeviceSynchronize());      
      
    const int iterations = 100;      
      
    // Unfused timing      
    CUDA_CHECK(cudaEventRecord(start));      
      
    for (int i = 0; i < iterations; ++i) {      
      
        softmax_kernel      
            <<<blocks, threads, scratch_shmem_size>>>(      
                d_X,      
                d_Y_temp,      
                N,      
                D      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
      
        layernorm_kernel      
            <<<blocks, threads, scratch_shmem_size>>>(      
                d_Y_temp,      
                d_Z_unfused,      
                d_gamma,      
                d_beta,      
                N,      
                D,      
                1e-5f      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
    }      
      
    CUDA_CHECK(cudaEventRecord(stop));      
    CUDA_CHECK(cudaEventSynchronize(stop));      
      
    float ms_unfused = 0.0f;      
      
    CUDA_CHECK(      
        cudaEventElapsedTime(      
            &ms_unfused,      
            start,      
            stop      
        )      
    );      
      
    ms_unfused /= iterations;      
      
    // Fused timing      
    CUDA_CHECK(cudaEventRecord(start));      
      
    for (int i = 0; i < iterations; ++i) {      
      
        fused_softmax_layernorm_kernel      
            <<<blocks, threads, fused_shmem_size>>>(      
                d_X,      
                d_Z_fused,      
                d_gamma,      
                d_beta,      
                N,      
                D,      
                1e-5f      
            );      
      
        CUDA_POST_KERNEL_CHECK();      
    }      
      
    CUDA_CHECK(cudaEventRecord(stop));      
    CUDA_CHECK(cudaEventSynchronize(stop));      
      
    float ms_fused = 0.0f;      
      
    CUDA_CHECK(      
        cudaEventElapsedTime(      
            &ms_fused,      
            start,      
            stop      
        )      
    );      
      
    ms_fused /= iterations;      
      
    CUDA_CHECK(      
        cudaMemcpy(      
            h_out_unfused.data(),      
            d_Z_unfused,      
            size_X,      
            cudaMemcpyDeviceToHost      
        )      
    );      
      
    CUDA_CHECK(      
        cudaMemcpy(      
            h_out_fused.data(),      
            d_Z_fused,      
            size_X,      
            cudaMemcpyDeviceToHost      
        )      
    );      
      
    double max_err = 0.0;      
    double l1_err = 0.0;      
      
    for (size_t i = 0; i < num_elements; ++i) {      
      
        double diff =      
            std::abs(      
                static_cast<double>(      
                    h_out_unfused[i]      
                ) -      
                static_cast<double>(      
                    h_out_fused[i]      
                )      
            );      
      
        max_err =      
            std::max(max_err, diff);      
      
        l1_err += diff;      
    }      
      
    double avg_err =      
        l1_err /      
        static_cast<double>(num_elements);      
      
    double speedup =      
        static_cast<double>(ms_unfused) /      
        static_cast<double>(ms_fused);      
      
    std::cout      
        << std::fixed      
        << std::setprecision(6);      
      
    std::cout      
        << "Unfused Kernel Pass Avg Latency: "      
        << ms_unfused      
        << " ms\n";      
      
    std::cout      
        << "Fused Operator Pass Avg Latency: "      
        << ms_fused      
        << " ms\n";      
      
    std::cout      
        << "Speedup: "      
        << speedup      
        << "x\n";      
      
    std::cout      
        << "Verification Max Absolute Error: "      
        << max_err      
        << "\n";      
      
    std::cout      
        << "Verification Mean L1 Error:     "      
        << avg_err      
        << "\n";      
      
    if (max_err < 1e-5) {      
        std::cout      
            << "Validation Check: PASS\n";      
    } else {      
        std::cout     
