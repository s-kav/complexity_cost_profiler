Bash
python examples/hpc_precision_demo.py 2>&1
complexity_cost_profiler — HPC / Mixed-Precision Demo
Python path: /home/user/complexity_cost_profiler/src
Show less
======================================================================
SECTION 1 — Precision-Accuracy Tradeoff Table
Base: FP64 algorithm with CU=200, EU=2.0 J, CO2=0.154 g, $=0.0008
======================================================================
Precision   Bits   Speedup   Enrg saved%  CO2 saved%  Acc loss ×     Machine ε   Dyn.range        CU    EU (J)
--------------------------------------------------------------------------------------------------------------
FP64          64       1.0           0.0         0.0         1.0      2.22e-16         307    200.00  2.000000
FP32          32       2.0          45.0        45.0  536936936.9      1.19e-07          38    100.00  1.100000
BF16          16       8.0          80.0        80.0  35225225225225.2      7.81e-03          38     25.00  0.400000
FP16          16       8.0          82.0        82.0  4396396396396.4      9.77e-04           4     25.00  0.360000
INT8           8      16.0          90.0        90.0  36036036036036.0      1.00e+00           2     12.50  0.200000

Key insight: FP16/BF16 deliver 8× speedup + ~82% energy savings,
but accuracy degrades by ~4 400× vs FP64 — only acceptable for
ML training with loss scaling or tasks that tolerate ~0.05% error.

======================================================================
SECTION 2 — Per-Precision Cost Analysis of an HPC Function
Function: dense matrix–vector product (n=64), MIXED_PRECISION profile
======================================================================

FP64 baseline metrics:
  CU=114.0000  EU=0.010180  CO2=0.00345800  $=0.00113900

Precision        CU        EU    Score  Grade     ε (machine)  Recommendation
--------------------------------------------------------------------------------
FP64      114.0000  0.010180    95.27  A+          2.22e-16  Recommended: FP64 offers 1.0x speedup and 0% energy reduction with acceptable accuracy.
FP32       57.0000  0.005599    97.65  A+          1.19e-07  Recommended: FP32 offers 2.0x speedup and 45% energy reduction with acceptable accuracy.
BF16       14.2500  0.002036    99.43  A+          7.81e-03  DL-friendly: BF16 gives 8.0x speedup — suitable for deep learning training (gradient averaging compensates low precision), not for iterative scientific solvers.
FP16       14.2500  0.001832    99.43  A+          9.77e-04  Conditional: FP16 gives 8.0x speedup but requires loss scaling or validation for your specific workload.
INT8        7.1250  0.001018    99.74  A+          1.00e+00  Inference-only: INT8 is 16.0x faster but requires explicit quantization — not a drop-in replacement for FP compute.

  Best for performance : INT8
  Best for energy      : INT8
  Best composite score : INT8

======================================================================
SECTION 3 — Pairwise Precision Comparison (FP64 as reference)
======================================================================

  FP64 → FP32
    Speedup            : 2.0×
    Energy reduction   : 45.0%
    CO₂ reduction      : 45.0%
    Cost reduction     : 45.0%
    Accuracy loss      : 536936936.9× (ε: 2.22e-16 → 1.19e-07)
    Dynamic range      : 307 → 38 decades
    → Recommended: FP32 offers 2.0x speedup and 45% energy reduction with acceptable accuracy.

  FP64 → FP16
    Speedup            : 8.0×
    Energy reduction   : 82.0%
    CO₂ reduction      : 82.0%
    Cost reduction     : 82.0%
    Accuracy loss      : 4396396396396.4× (ε: 2.22e-16 → 9.77e-04)
    Dynamic range      : 307 → 4 decades
    ⚠ WARNING: Limited dynamic range: 4 decades (down from 307) — overflow/underflow risk.
    ⚠ WARNING: Moderate precision (ε≈4.9e-04) — validate with loss scaling or mixed-precision accumulation before production use.
    → Conditional: FP16 gives 8.0x speedup but requires loss scaling or validation for your specific workload.

  FP64 → BF16
    Speedup            : 8.0×
    Energy reduction   : 80.0%
    CO₂ reduction      : 80.0%
    Cost reduction     : 80.0%
    Accuracy loss      : 35225225225225.2× (ε: 2.22e-16 → 7.81e-03)
    Dynamic range      : 307 → 38 decades
    ⚠ WARNING: Low precision (ε≈3.9e-03): suitable for DL training with gradient averaging, but not for iterative scientific HPC kernels.
    → DL-friendly: BF16 gives 8.0x speedup — suitable for deep learning training (gradient averaging compensates low precision), not for iterative scientific solvers.

  FP64 → INT8
    Speedup            : 16.0×
    Energy reduction   : 90.0%
    CO₂ reduction      : 90.0%
    Cost reduction     : 90.0%
    Accuracy loss      : 36036036036036.0× (ε: 2.22e-16 → 1.00e+00)
    Dynamic range      : 307 → 2 decades
    ⚠ WARNING: Limited dynamic range: 2 decades (down from 307) — overflow/underflow risk.
    ⚠ WARNING: Integer format — max relative error ≈4.0e-03. Requires explicit quantization (calibration, zero-point, scale).
    ⚠ WARNING: INT8 requires explicit quantization (calibration, zero-point, scale). Cannot be used as a drop-in replacement.
    → Inference-only: INT8 is 16.0x faster but requires explicit quantization — not a drop-in replacement for FP compute.

  FP32 → FP16
    Speedup            : 4.0×
    Energy reduction   : 67.3%
    CO₂ reduction      : 67.3%
    Cost reduction     : 67.3%
    Accuracy loss      : 8187.9× (ε: 1.19e-07 → 9.77e-04)
    Dynamic range      : 38 → 4 decades
    ⚠ WARNING: Limited dynamic range: 4 decades (down from 38) — overflow/underflow risk.
    ⚠ WARNING: Moderate precision (ε≈4.9e-04) — validate with loss scaling or mixed-precision accumulation before production use.
    → Conditional: FP16 gives 4.0x speedup but requires loss scaling or validation for your specific workload.

======================================================================
SECTION 4 — Task-Specific Precision Recommendations
======================================================================

  Task: Climate / CFD simulation
    Min required precision: FP64  (Accumulation errors over thousands of timesteps require FP64)
    Viable options (all meet min-precision requirement), lowest CU first:
      INT8: CU=31.25  EU=0.5000  ε=4.00e-03 ← fastest viable option
      BF16: CU=62.50  EU=1.0000  ε=3.91e-03
      FP16: CU=62.50  EU=0.9000  ε=4.88e-04
      FP32: CU=250.00  EU=2.7500  ε=5.96e-08
      FP64: CU=500.00  EU=5.0000  ε=1.11e-16

  Task: Iterative linear solver
    Min required precision: FP32  (Mixed-precision iterative refinement converges with FP32 base)
    Viable options (all meet min-precision requirement), lowest CU first:
      INT8: CU=31.25  EU=0.5000  ε=4.00e-03 ← fastest viable option
      BF16: CU=62.50  EU=1.0000  ε=3.91e-03
      FP16: CU=62.50  EU=0.9000  ε=4.88e-04
      FP32: CU=250.00  EU=2.7500  ε=5.96e-08

  Task: Deep learning training
    Min required precision: FP16  (Mixed precision with loss scaling is standard practice)
    Viable options (all meet min-precision requirement), lowest CU first:
      INT8: CU=31.25  EU=0.5000  ε=4.00e-03 ← fastest viable option
      FP16: CU=62.50  EU=0.9000  ε=4.88e-04

  Task: DL inference (quantized)
    Min required precision: INT8  (Post-training quantization typically works well)
    Viable options (all meet min-precision requirement), lowest CU first:
      INT8: CU=31.25  EU=0.5000  ε=4.00e-03 ← fastest viable option

  Task: Signal processing / FFT
    Min required precision: FP32  (FFT at FP32 sufficient for most signal analysis)
    Viable options (all meet min-precision requirement), lowest CU first:
      INT8: CU=31.25  EU=0.5000  ε=4.00e-03 ← fastest viable option
      BF16: CU=62.50  EU=1.0000  ε=3.91e-03
      FP16: CU=62.50  EU=0.9000  ε=4.88e-04
      FP32: CU=250.00  EU=2.7500  ε=5.96e-08

======================================================================
SECTION 5 — HPC Library Benchmark
Operations: matrix_multiply, fft   |   Profile: MIXED_PRECISION
======================================================================
Op               Library  Prec      N       ms       CU   Score Grade
---------------------------------------------------------------------
fft              numpy    FP64     64     0.03   0.0002   99.97   A+
fft              numpy    FP32     64     0.03   0.0001   99.98   A+
fft              numpy    FP64    128     0.11   0.0011   99.90   A+
fft              numpy    FP32    128     0.11   0.0006   99.94   A+
fft              numpy    FP64    256     1.02   0.0052   99.04   A+
fft              numpy    FP32    256     1.07   0.0026   99.45   A+
fft              scipy    FP64     64     0.02   0.0002   99.98   A+
fft              scipy    FP32     64     0.02   0.0001   99.99   A+
fft              scipy    FP64    128     0.06   0.0011   99.94   A+
fft              scipy    FP32    128     0.06   0.0006   99.97   A+
fft              scipy    FP64    256     0.49   0.0052   99.54   A+
fft              scipy    FP32    256     0.26   0.0026   99.87   A+
matrix_multiply  numpy    FP64     64     0.01   0.0005   99.99   A+
matrix_multiply  numpy    FP32     64     0.00   0.0003  100.00   A+
matrix_multiply  numpy    FP64    128     0.20   0.0042   99.82   A+
matrix_multiply  numpy    FP32    128     0.08   0.0021   99.96   A+
matrix_multiply  numpy    FP64    256     0.48   0.0336   99.55   A+
matrix_multiply  numpy    FP32    256     0.16   0.0168   99.92   A+
matrix_multiply  scipy    FP64     64     0.01   0.0005   99.99   A+
matrix_multiply  scipy    FP32     64     0.00   0.0003  100.00   A+
matrix_multiply  scipy    FP64    128     0.07   0.0042   99.93   A+
matrix_multiply  scipy    FP32    128     0.06   0.0021   99.97   A+
matrix_multiply  scipy    FP64    256     0.25   0.0336   99.77   A+
matrix_multiply  scipy    FP32    256     0.16   0.0168   99.92   A+

  Best for 'matrix_multiply' (n=128): library=scipy  precision=FP32  score=99.96 (A+)  elapsed=0.08 ms
  Best for 'fft' (n=128): library=scipy  precision=FP32  score=99.97 (A+)  elapsed=0.06 ms

======================================================================
SECTION 6 — PTX Analysis with Explicit Precision Override
File: matrixMul_kernel_32.ptx
======================================================================

Precision        CU          EU    Score  Grade
--------------------------------------------------
FP64      504.0000    0.052100    74.07  B
FP32      337.0000    0.037070    82.64  A-
FP16      211.7500    0.024712    89.08  A

======================================================================
Demo complete.
======================================================================

## New files (4 files)

| File | Description |
| :--- | :--- |
| `cost_models/precision_profiles.json` | Precision Specs: multipliers для CU/EU/CO2/$, machine epsilon, dynamic range для FP64/FP32/BF16/FP16/INT8. Основа — NVIDIA A100 + IEEE 754 |
| `src/precision_accuracy_model.py` | Tradeoff Analysis Module: `adjust_metrics()`, `compare_precisions()`, `build_tradeoff_table()`, `recommend_precision()` |
| `src/hpc_library_integration.py` | `HPCLibraryProfiler` — benchmarks GEMM/FFT/solve/SVD/norm для NumPy/SciPy/CuPy with mapping `elapsed` → CU/EU/CO2/$ |
| `examples/hpc_precision_demo.py` | Demo script: 6 sections, shows tradeoff tables, analysis of functions by accuracy, comparison of libraries, PTX |

## Changed files (5 files)

| File | Changes |
| :--- | :--- |
| `cost_models/gpu_ptx_instr_costs.json` | Added instructions for all formats: ADD.F64/F32/F16, MUL.F64/F32/F16, FMA.\*, HMMA.F16, DP4A, IMAD.S8, LD/ST.GLOBAL.F64/F32/F16 |
| `cost_models/config_weights.json` | Profile added `mixed_precision` (CU=0.40, EU=0.35, CO2=0.15, $=0.10) |
| `src/instruction_cost_model.py` | Added `PRECISION_MULTIPLIERS`, parameter `precision=` в `get_cost()`, autodetect format from PTX instruction suffix (MUL.F16 → FP16), method `get_precision_multipliers()` |
| `src/enhanced_cost_analyzer.py` | 3 new methods: `analyze_function_with_precision()`, `compare_precisions()`, `analyze_ptx_with_precision()` |
| `src/composite_score_calculator.py` | Profile exported `MIXED_PRECISION` in `PROFILE_WEIGHTS`, updated `load_configuration()` и `WeightConfiguration` |

======================================================================

What was generated:

| File | Description |
| :--- | :--- |
| `fig1_pareto_front` | Pareto front: speedup vs. accuracy loss (log-log scatter) |
| `fig2_cost_reduction` | Grouped bar chart: Energy/CO₂/$ savings by precision format |
| `fig3_composite_score` | Composite efficiency score by precision format (with grades A+ … F) |
| `fig4_library_benchmark` | Execution time of GEMM and FFT vs. problem size (NumPy/SciPy) |
| `fig5_task_heatmap` | Task × precision compatibility heatmap |

======================================================================