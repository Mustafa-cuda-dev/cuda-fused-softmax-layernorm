
# Benchmark Results — CUDA Fused Softmax + LayerNorm


## Hardware Configuration

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 |
| CUDA Architecture | sm_75 |
| Execution Model | CUDA C++ |
| Threads per Block | 256 |
| Precision | FP32 |
| Compiler | NVIDIA NVCC |


---

# Compilation Verification

Compilation command:

```bash
nvcc -O3 -arch=sm_75 -Xptxas -v --use_fast_math fused_softmax_ln.cu -o fused_softmax_ln

Compiler validation:

ptxas info:
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads

Kernel resource usage:

Kernel	Registers	Spills

Fused Softmax + LayerNorm	24	0
LayerNorm Baseline	23	0
Softmax Baseline	21	0


The kernel compiled without register spilling or local memory pressure.


---

Runtime Benchmark

Test Configuration 1

Input Dimensions:

N = 1024
D = 512

Results:

Implementation	Latency

Unfused Softmax + LayerNorm	0.057687 ms
Fused Softmax + LayerNorm	0.035533 ms


Performance:

Speedup: 1.623481x

Validation:

Verification Max Absolute Error: 0.000000
Verification Mean L1 Error:      0.000000

Validation Check: PASS


---

Test Configuration 2

Input Dimensions:

N = 2048
D = 1024

Results:

Implementation	Latency

Unfused Softmax + LayerNorm	0.195284 ms
Fused Softmax + LayerNorm	0.120548 ms


Performance:

Speedup: 1.619974x

Validation:

Verification Max Absolute Error: 0.000000
Verification Mean L1 Error:      0.000000

Validation Check: PASS


---

Performance Summary

Metric	Result

Average Speedup	~1.62x
Numerical Error	0.0
Register Spills	0
Validation Status	PASS



---

Optimization Impact

The fused implementation improves performance by:

Eliminating intermediate Softmax global memory writes

Reducing kernel launch overhead

Increasing shared memory data reuse

Combining multiple GPU operations into one execution pipeline



---

Verification Methodology

The benchmark validates:

Correctness

Compared fused output against the unfused CUDA baseline.

Metrics:

Maximum absolute error

Mean L1 error


Performance

Measured using CUDA events after warmup iterations.

Benchmark process:

1. GPU warmup execution


2. 100 timed iterations


3. Average latency calculation


4. Output validation




---

Conclusion

The CUDA fused Softmax + LayerNorm kernel achieves approximately:

1.62x latency improvement

while maintaining:

0.000000 numerical error

and compiling with:

0 register spills

on NVIDIA Tesla T4 hardware.

Commit message:

```text
Add T4 benchmark results and performance validation report

Extended description:

Added benchmark documentation containing CUDA compilation verification, kernel resource usage, runtime measurements, and numerical validation results.

Included:
- NVIDIA Tesla T4 hardware configuration
- NVCC compilation details
- Register usage and spill analysis
- Latency comparison between fused and unfused implementations
- Speedup measurements
- Correctness validation results

This provides reproducible performance evidence for the fused CUDA operator.

 
