
# Fused CUDA Softmax + LayerNorm Kernel

A high-performance CUDA implementation that fuses **Softmax + LayerNorm** into a single GPU kernel to reduce intermediate memory traffic and improve latency.

The project focuses on GPU performance optimization techniques including:
- Kernel fusion
- Shared memory reuse
- Warp-level reduction primitives
- Dynamic shared memory management
- Numerical correctness validation

---

## Motivation

Modern deep learning workloads frequently execute Softmax and LayerNorm as separate GPU operations.

Traditional pipeline:

Input | v Softmax Kernel | v Global Memory Write (HBM) | v LayerNorm Kernel | v Output

This introduces unnecessary:
- Global memory traffic
- Kernel launch overhead
- Intermediate tensor storage

This project combines both operations into a single fused CUDA kernel:

Input | v Fused Softmax + LayerNorm Kernel | v Output

The intermediate Softmax output remains inside fast on-chip shared memory instead of being written back to HBM.

---

#  Key Optimizations

## 1. CUDA Kernel Fusion

Combined:

Softmax → LayerNorm

into one GPU kernel.

Benefits:
- Eliminates intermediate global memory round-trip
- Reduces kernel launch overhead
- Improves memory efficiency

---

## 2. Shared Memory Reuse

The fused kernel stores intermediate Softmax values in dynamic shared memory:

Shared Memory Layout:

+----------------+ | Reduction      | | Scratch Space  | +----------------+ | Softmax Values | | (s_y buffer)   | +----------------+

This avoids unnecessary HBM access.

---

## 3. Warp-Level Reductions

Implemented custom block reductions using CUDA warp shuffle primitives:

- `__shfl_down_sync`
- Warp cooperative reduction
- Shared-memory warp aggregation

Used for:

- Maximum reduction
- Sum reduction
- Variance calculation

---

## 4. Correctness-Safe Synchronization

Implemented synchronization protocols to avoid shared-memory hazards:

- Cross-warp WAR hazard prevention
- Safe shared-memory reuse
- Deterministic reduction ordering

---

# 🖥️ Hardware Tested

## NVIDIA Tesla T4

Architecture: Turing Compute Capability: sm_75 Precision: FP32 CUDA Kernel Compilation: nvcc -O3

---

#  Benchmark Results

## Test 1

Configuration:

N = 1024 D = 512

| Kernel | Latency |
|---|---:|
| Unfused Softmax + LayerNorm | 0.057687 ms |
| Fused CUDA Kernel | 0.035533 ms |

### Speedup

1.62x Faster

---

## Test 2

Configuration:

N = 2048 D = 1024

| Kernel | Latency |
|---|---:|
| Unfused Softmax + LayerNorm | 0.195284 ms |
| Fused CUDA Kernel | 0.120548 ms |

### Speedup

1.62x Faster

---

# Correctness Validation

Compared fused kernel output against the unfused CUDA baseline.

Results:

Maximum Absolute Error: 0.000000 Mean L1 Error:           0.000000

Validation: PASS

The fused implementation produces numerically equivalent results under tested configurations.

---

#  Build Instructions

Compile:

```bash
nvcc -O3 \
-arch=sm_75 \
-Xptxas -v \
--use_fast_math \
fused_softmax_ln.cu \
-o fused_softmax_ln

Run:

./fused_softmax_ln


---

 Project Structure

cuda-fused-softmax-layernorm/

├── fused_softmax_ln.cu
├── README.md
└── LICENSE


---

 Technical Highlights

Custom CUDA reduction primitives

Dynamic shared memory allocation

Warp shuffle optimization

Memory traffic reduction

GPU kernel fusion

FP32 numerical validation

Performance benchmarking



---

Future Improvements

Possible extensions:

Tensor Core acceleration

FP16/BF16 support

Persistent kernel design

Nsight Compute profiling

Support for larger transformer dimensions

Integration with PyTorch CUDA extensions



---

Author

CUDA Engineer | GPU Performance Optimization

Focused on building high-performance GPU kernels for deep learning workloads.

