# complexity_cost_profiler/examples/hpc_precision_demo.py
#!/usr/bin/env python3

"""
HPC Library Integration & Mixed-Precision Analysis Demo
========================================================
 
Demonstrates three new capabilities added to complexity_cost_profiler:
 
  1. Mixed-precision cost analysis
     Show how CU / EU / CO2 / $ change when we switch from FP64 → FP32 →
     FP16 → INT8 for a representative HPC function.
 
  2. Precision-accuracy tradeoff table
     For every precision format, print the speedup factor, energy savings,
     and accuracy loss factor relative to FP64, so the reader can see the
     price-performance-accuracy surface at a glance.
 
  3. HPC library comparison  (NumPy / SciPy / CuPy if available)
     Profile matrix multiplication, FFT, and a linear solve across all
     installed libraries and recommend the best choice per task.
 
Run from the repository root:
 
    cd complexity_cost_profiler
    PYTHONPATH=src python examples/hpc_precision_demo.py
 
Optional flags printed to stdout; no file I/O required.
"""
 
import os
import sys
 
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(os.path.join(_ROOT))           # cost_models/ must be reachable
 
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from precision_accuracy_model   import PrecisionAccuracyModel, SUPPORTED_PRECISIONS
from enhanced_cost_analyzer     import EnhancedCostAnalyzer
from hpc_library_integration    import HPCLibraryProfiler
 
 
# ===========================================================================
# Section 1 — Precision-accuracy tradeoff table
# ===========================================================================
 
def demo_tradeoff_table() -> None:
    print("\n" + "=" * 70)
    print("SECTION 1 — Precision-Accuracy Tradeoff Table")
    print("Base: FP64 algorithm with CU=200, EU=2.0 J, CO2=0.154 g, $=0.0008")
    print("=" * 70)
 
    base = {"CU": 200.0, "EU": 2.0, "CO2": 0.000154, "$": 0.0008}
    model = PrecisionAccuracyModel()
    table = model.build_tradeoff_table(base, base_precision="FP64")
 
    # Header
    cols = [
        ("Precision",        "<10"),
        ("Bits",             ">4"),
        ("Speedup",          ">8"),
        ("Enrg saved%",      ">12"),
        ("CO2 saved%",       ">10"),
        ("Acc loss ×",       ">10"),
        ("Machine ε",        ">12"),
        ("Dyn.range",        ">10"),
        ("CU",               ">8"),
        ("EU (J)",           ">8"),
    ]
    header = "  ".join(f"{h:{f}}" for h, f in cols)
    print(header)
    print("-" * len(header))
 
    for row in table:
        print(
            f"{row['Precision']:<10}  "
            f"{row['Bits']:>4}  "
            f"{row['Speedup_vs_base']:>8.1f}  "
            f"{row['Energy_saved_%']:>12.1f}  "
            f"{row['CO2_saved_%']:>10.1f}  "
            f"{row['Accuracy_loss_factor']:>10.1f}  "
            f"{row['Machine_epsilon']:>12.2e}  "
            f"{row['Dynamic_range_dec']:>10.0f}  "
            f"{row['CU']:>8.2f}  "
            f"{row['EU']:>8.6f}"
        )
 
    print()
    print("Key insight: FP16/BF16 deliver 8× speedup + ~82% energy savings,")
    print("but accuracy degrades by ~4 400× vs FP64 — only acceptable for")
    print("ML training with loss scaling or tasks that tolerate ~0.05% error.")
 
 
# ===========================================================================
# Section 2 — Per-precision cost analysis of an HPC function
# ===========================================================================
 
def _sample_hpc_function():
    """Matrix–vector product simulation (pure Python for bytecode analysis)."""
    n = 64
    A = [[float(i + j) for j in range(n)] for i in range(n)]
    b = [float(i) for i in range(n)]
    result = []
    for row in A:
        s = 0.0
        for a_ij, b_j in zip(row, b):
            s += a_ij * b_j
        result.append(s)
    return result
 
 
def demo_function_precision_analysis() -> None:
    print("\n" + "=" * 70)
    print("SECTION 2 — Per-Precision Cost Analysis of an HPC Function")
    print("Function: dense matrix–vector product (n=64), MIXED_PRECISION profile")
    print("=" * 70)
 
    analyzer = EnhancedCostAnalyzer(arch="x86_64", profile="MIXED_PRECISION")
    comparison = analyzer.compare_precisions(
        _sample_hpc_function,
        precisions=SUPPORTED_PRECISIONS,
        include_composite=True,
    )
 
    base = comparison["base_metrics"]
    print("\nFP64 baseline metrics:")
    print(f"  CU={base['CU']:.4f}  EU={base['EU']:.6f}  "
          f"CO2={base['CO2']:.8f}  $={base['$']:.8f}")
 
    print(f"\n{'Precision':<8}  {'CU':>8}  {'EU':>8}  {'Score':>7}  {'Grade':<4}  "
          f"{'ε (machine)':>14}  Recommendation")
    print("-" * 80)
 
    model = PrecisionAccuracyModel()
    for precision, metrics in comparison["by_precision"].items():
        spec = model.get_spec(precision)
        tradeoff = model.compare_precisions("FP64", precision)
        print(
            f"{precision:<8}  "
            f"{metrics.get('CU', 0):>8.4f}  "
            f"{metrics.get('EU', 0):>8.6f}  "
            f"{metrics.get('COMPOSITE_SCORE', 0):>7.2f}  "
            f"{metrics.get('SCORE_GRADE', '?'):<4}  "
            f"{spec.machine_epsilon:>14.2e}  "
            f"{tradeoff.recommendation}"
        )
 
    print(f"\n  Best for performance : {comparison['best_for_performance']}")
    print(f"  Best for energy      : {comparison['best_for_energy']}")
    print(f"  Best composite score : {comparison['best_balanced']}")
 
 
# ===========================================================================
# Section 3 — Pairwise precision comparison narrative
# ===========================================================================
 
def demo_pairwise_comparisons() -> None:
    print("\n" + "=" * 70)
    print("SECTION 3 — Pairwise Precision Comparison (FP64 as reference)")
    print("=" * 70)
 
    model = PrecisionAccuracyModel()
    pairs = [
        ("FP64", "FP32"),
        ("FP64", "FP16"),
        ("FP64", "BF16"),
        ("FP64", "INT8"),
        ("FP32", "FP16"),
    ]
 
    for src, dst in pairs:
        t = model.compare_precisions(src, dst)
        print(f"\n  {src} → {dst}")
        print(f"    Speedup            : {t.speedup:.1f}×")
        print(f"    Energy reduction   : {t.energy_reduction_pct:.1f}%")
        print(f"    CO₂ reduction      : {t.co2_reduction_pct:.1f}%")
        print(f"    Cost reduction     : {t.cost_reduction_pct:.1f}%")
        print(f"    Accuracy loss      : {t.accuracy_loss_factor:.1f}× "
              f"(ε: {t.epsilon_from:.2e} → {t.epsilon_to:.2e})")
        print(f"    Dynamic range      : {t.dynamic_range_from:.0f} → "
              f"{t.dynamic_range_to:.0f} decades")
        if t.warnings:
            for w in t.warnings:
                print(f"    ⚠ WARNING: {w}")
        print(f"    → {t.recommendation}")
 
 
# ===========================================================================
# Section 4 — Task-specific precision recommendation
# ===========================================================================
 
def demo_task_recommendations() -> None:
    print("\n" + "=" * 70)
    print("SECTION 4 — Task-Specific Precision Recommendations")
    print("=" * 70)
 
    model = PrecisionAccuracyModel()
    base  = {"CU": 500.0, "EU": 5.0, "CO2": 0.000384, "$": 0.002}
 
    tasks = [
        ("climate_simulation",    "Climate / CFD simulation"),
        ("iterative_solver",      "Iterative linear solver"),
        ("deep_learning_training","Deep learning training"),
        ("dl_inference",          "DL inference (quantized)"),
        ("signal_processing_fft", "Signal processing / FFT"),
    ]
 
    for task_key, task_label in tasks:
        candidates = model.recommend_precision(base, task_type=task_key)
        req = model.task_requirements.get(task_key, {})
        min_p = req.get("min_precision", "FP64")
        justification = req.get("justification", "")
 
        print(f"\n  Task: {task_label}")
        print(f"    Min required precision: {min_p}  ({justification})")
        print("    Viable options (all meet min-precision requirement), lowest CU first:")
        sorted_cands = sorted(candidates.items(),
                              key=lambda kv: kv[1].adjusted_metrics["CU"])
        for i, (prec, adj) in enumerate(sorted_cands):
            tag = " ← fastest viable option" if i == 0 else ""
            print(f"      {prec}: CU={adj.adjusted_metrics['CU']:.2f}  "
                  f"EU={adj.adjusted_metrics['EU']:.4f}  "
                  f"ε={adj.accuracy_info['max_relative_error']:.2e}{tag}")
 
 
# ===========================================================================
# Section 5 — HPC library benchmark (NumPy + SciPy; CuPy if available)
# ===========================================================================
 
def demo_hpc_library_comparison() -> None:
    print("\n" + "=" * 70)
    print("SECTION 5 — HPC Library Benchmark")
    print("Operations: matrix_multiply, fft   |   Profile: MIXED_PRECISION")
    print("=" * 70)
 
    profiler = HPCLibraryProfiler(
        sizes=[64, 128, 256],
        precisions=["FP64", "FP32"],
        repeats=3,
        composite_profile="MIXED_PRECISION",
    )
 
    results = profiler.profile_all(operations=["matrix_multiply", "fft"])
 
    if not results:
        print("  No results — NumPy not installed?")
        return
 
    # Print compact table
    header = (f"{'Op':<16} {'Library':<8} {'Prec':<6} {'N':>4} "
              f"{'ms':>8} {'CU':>8} {'Score':>7} {'Grade':<4}")
    print(header)
    print("-" * len(header))
 
    for r in sorted(results,
                    key=lambda x: (x.get("operation",""), x.get("library",""),
                                   x.get("size", 0))):
        ms = r.get("elapsed_ms", float("nan"))
        if ms != ms:           # nan
            continue
        print(
            f"{r.get('operation',''):<16} "
            f"{r.get('library',''):<8} "
            f"{r.get('precision',''):<6} "
            f"{r.get('size',0):>4} "
            f"{ms:>8.2f} "
            f"{r.get('CU', 0):>8.4f} "
            f"{r.get('COMPOSITE_SCORE', 0):>7.2f} "
            f"{r.get('SCORE_GRADE',''):>4}"
        )
 
    print()
    # Per-operation recommendation
    for op in ["matrix_multiply", "fft"]:
        best = profiler.recommend_library(operation=op, size=128)
        if best:
            print(f"  Best for '{op}' (n=128): "
                  f"library={best.get('library')}  "
                  f"precision={best.get('precision')}  "
                  f"score={best.get('composite_score', 0):.2f} "
                  f"({best.get('score_grade','?')})  "
                  f"elapsed={best.get('elapsed_ms', 0):.2f} ms")
 
 
# ===========================================================================
# Section 6 — PTX precision analysis
# ===========================================================================
 
def demo_ptx_precision() -> None:
    ptx_path = os.path.join(_ROOT, "examples", "matrixMul_kernel_32.ptx")
    if not os.path.exists(ptx_path):
        return
 
    print("\n" + "=" * 70)
    print("SECTION 6 — PTX Analysis with Explicit Precision Override")
    print(f"File: {os.path.basename(ptx_path)}")
    print("=" * 70)
 
    analyzer = EnhancedCostAnalyzer(arch="gpu_ptx", profile="HPC")
 
    print(f"\n{'Precision':<8}  {'CU':>8}  {'EU':>10}  {'Score':>7}  {'Grade'}")
    print("-" * 50)
 
    for precision in ["FP64", "FP32", "FP16"]:
        result = analyzer.analyze_ptx_with_precision(
            ptx_path, precision=precision, include_composite=True
        )
        print(
            f"{precision:<8}  "
            f"{result.get('CU', 0):>8.4f}  "
            f"{result.get('EU', 0):>10.6f}  "
            f"{result.get('COMPOSITE_SCORE', 0):>7.2f}  "
            f"{result.get('SCORE_GRADE', '?')}"
        )
 
 
# ===========================================================================
# Main
# ===========================================================================
 
if __name__ == "__main__":
    print("complexity_cost_profiler — HPC / Mixed-Precision Demo")
    print("Python path:", sys.path[0])
 
    demo_tradeoff_table()
    demo_function_precision_analysis()
    demo_pairwise_comparisons()
    demo_task_recommendations()
    demo_hpc_library_comparison()
    demo_ptx_precision()
 
    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)