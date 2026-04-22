# complexity_cost_profiler/src/hpc_library_integration.py

"""
HPC Library Integration
 
Profiles common HPC operations across multiple libraries and maps measured
wall-clock performance to CCP cost metrics (CU, EU, CO2, $).
 
Supported libraries (all optional — the profiler degrades gracefully):
  - NumPy   (CPU BLAS/LAPACK backend, always available)
  - SciPy   (optimised CPU routines for FFT, linear-algebra, etc.)
  - CuPy    (GPU drop-in for NumPy — requires CUDA)
 
Supported operations:
  - matrix_multiply  — dense GEMM
  - fft              — 1-D Fast Fourier Transform
  - linear_solve     — solve Ax = b (direct LU)
  - svd              — Singular Value Decomposition
  - norm             — vector / matrix norm
 
For each (library, precision, operation, size) combination the profiler:
  1. Warms up (one dry run)
  2. Measures median wall-clock time over several repetitions
  3. Converts elapsed time → CU / EU / CO2 / $ via configurable hardware spec
  4. Calls PrecisionAccuracyModel to attach accuracy metadata
  5. Runs CompositeScoreCalculator to produce a composite score
 
The output is a list of result dicts suitable for pandas.DataFrame.
"""
 
import time
import os
import math
import importlib
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple
 
_HERE = os.path.dirname(os.path.abspath(__file__))
 
# ---------------------------------------------------------------------------
# Lazy imports — libraries are optional
# ---------------------------------------------------------------------------
 
def _try_import(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
 
 
np   = _try_import("numpy")
sp   = _try_import("scipy")
cupy = _try_import("cupy")
 
if sp is not None:
    sp_fft    = _try_import("scipy.fft")
    sp_linalg = _try_import("scipy.linalg")
else:
    sp_fft    = None
    sp_linalg = None
 
 
# ---------------------------------------------------------------------------
# Hardware performance model
# Converts elapsed seconds → abstract cost metrics.
#
# Baseline: CPU node with 1 socket Intel Xeon Platinum 8380 (Ice Lake)
#   TDP  ≈ 270 W, peak FP64 ≈ 2.4 TFLOP/s (AVX-512 FMA)
# GPU node: NVIDIA A100 SXM4 80 GB
#   TDP  ≈ 400 W, peak FP64 ≈ 9.7 TFLOP/s
# Cloud pricing: AWS p3.2xlarge ~ $3.06/hr → ~$8.5e-4/s
# Carbon intensity: EU avg 0.276 kgCO2/kWh = 7.67e-5 kgCO2/s per 1W load
# ---------------------------------------------------------------------------
 
CPU_SPEC = {
    "peak_flops":    2.4e12,   # FP64 FLOP/s
    "tdp_watts":     270.0,
    "cloud_usd_per_s": 4.0e-4, # ~$1.44/hr spot CPU
    "co2_per_joule": 7.67e-5,  # kgCO2/J  (EU avg grid)
    "cu_per_flop":   1.0e-9,   # 1 CU = 1 GFLOP
}
 
GPU_SPEC = {
    "peak_flops":    9.7e12,
    "tdp_watts":     400.0,
    "cloud_usd_per_s": 8.5e-4,
    "co2_per_joule": 7.67e-5,
    "cu_per_flop":   1.0e-9,
}
 
HARDWARE_SPECS = {"cpu": CPU_SPEC, "gpu": GPU_SPEC}
 
 
def _elapsed_to_metrics(elapsed_s: float, flop_count: int, hw: str = "cpu") -> Dict[str, float]:
    """Convert elapsed seconds + flop count to CU/EU/CO2/$."""
    spec = HARDWARE_SPECS.get(hw, CPU_SPEC)
    cu   = flop_count * spec["cu_per_flop"]
    eu   = elapsed_s * spec["tdp_watts"]        # Joules (energy)
    co2  = eu * spec["co2_per_joule"]            # kg CO2
    usd  = elapsed_s * spec["cloud_usd_per_s"]
    return {"CU": cu, "EU": eu, "CO2": co2, "$": usd}
 
 
# ---------------------------------------------------------------------------
# FLOPs formulae for standard HPC kernels
# ---------------------------------------------------------------------------
 
def _flops_gemm(n: int, m: int = 0, k: int = 0) -> int:
    """Dense GEMM: C = A @ B, A:(n×k), B:(k×m) → 2nmk FLOPs."""
    if m == 0: m = n
    if k == 0: k = n
    return 2 * n * m * k
 
 
def _flops_fft(n: int) -> int:
    """Radix-2 FFT: ~5 n log2(n) FLOPs (complex)."""
    return max(1, int(5 * n * math.log2(max(n, 2))))
 
 
def _flops_linear_solve(n: int) -> int:
    """LU factorisation + triangular solve: 2/3 n³ + n² FLOPs."""
    return int((2 / 3) * n ** 3 + n ** 2)
 
 
def _flops_svd(m: int, n: int = 0) -> int:
    """Full SVD: ~4m²n + 8mn² + 9n³ for m>=n (Golub-Reinsch)."""
    if n == 0: n = m
    if m < n: m, n = n, m
    return int(4 * m * n ** 2 + 8 * m * n ** 2 + 9 * n ** 3)
 
 
def _flops_norm(n: int) -> int:
    """L2 vector norm: 2n FLOPs."""
    return 2 * n
 
 
# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
 
def _make_matrix(n: int, precision: str, xp: Any) -> Any:
    """Return an (n×n) random matrix in the requested precision."""
    dtype_map = {
        "FP64": "float64",
        "FP32": "float32",
        "FP16": "float16",
        "BF16": "bfloat16",  # CuPy supports bfloat16; NumPy does not natively
        "INT8": "int8",
    }
    dtype = dtype_map.get(precision, "float64")
    try:
        return xp.random.randn(n, n).astype(dtype)
    except (TypeError, AttributeError):
        # fall back to float32 if dtype unsupported
        return xp.random.randn(n, n).astype("float32")
 
 
def _make_vector(n: int, precision: str, xp: Any) -> Any:
    dtype_map = {
        "FP64": "float64", "FP32": "float32",
        "FP16": "float16", "BF16": "bfloat16", "INT8": "int8",
    }
    dtype = dtype_map.get(precision, "float64")
    try:
        return xp.random.randn(n).astype(dtype)
    except (TypeError, AttributeError):
        return xp.random.randn(n).astype("float32")
 
 
# ---------------------------------------------------------------------------
# Micro-benchmark runner
# ---------------------------------------------------------------------------
 
def _benchmark(func: Callable, repeats: int = 5) -> float:
    """Return median wall-clock time (seconds) over *repeats* calls."""
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
 
 
# ---------------------------------------------------------------------------
# HPCLibraryProfiler
# ---------------------------------------------------------------------------
 
class HPCLibraryProfiler:
    """
    Profile HPC operations across NumPy, SciPy, and CuPy and map results
    to CCP cost metrics (CU, EU, CO2, $) with precision-accuracy metadata.
 
    Parameters
    ----------
    sizes : list of int
        Problem sizes to benchmark (matrix dimension N or vector length).
    precisions : list of str
        Precision formats to test.  BF16 and FP16 are CPU-limited and may
        silently fall back to FP32 on NumPy.
    repeats : int
        Number of timed repetitions per (library, precision, op, size).
    composite_profile : str
        CCP profile name for composite scoring ('HPC', 'DEFAULT', …).
    """
 
    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        precisions: Optional[List[str]] = None,
        repeats: int = 5,
        composite_profile: str = "HPC",
    ) -> None:
        self.sizes     = sizes     or [64, 128, 256, 512]
        self.precisions = precisions or ["FP64", "FP32", "FP16"]
        self.repeats   = repeats
        self.composite_profile = composite_profile
 
        # Lazy-import project modules so this file can be loaded standalone
        import sys
        sys.path.insert(0, _HERE)
        from composite_score_calculator import CompositeScoreCalculator
        from precision_accuracy_model   import PrecisionAccuracyModel
 
        self._scorer     = CompositeScoreCalculator(profile=composite_profile)
        self._prec_model = PrecisionAccuracyModel()
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def profile_all(self, operations: Optional[List[str]] = None) -> List[Dict]:
        """
        Run all benchmarks and return a flat list of result rows.
 
        Parameters
        ----------
        operations : list of str, optional
            Subset of operations to run.  Available:
            'matrix_multiply', 'fft', 'linear_solve', 'svd', 'norm'.
            Defaults to all.
 
        Returns
        -------
        list of dict
            Each dict is one (library × precision × operation × size) row
            with metrics, accuracy info, and composite score.
        """
        available_ops = {
            "matrix_multiply": self._bench_gemm,
            "fft":             self._bench_fft,
            "linear_solve":    self._bench_linear_solve,
            "svd":             self._bench_svd,
            "norm":            self._bench_norm,
        }
        ops = operations or list(available_ops.keys())
 
        rows: List[Dict] = []
        for op_name in ops:
            if op_name not in available_ops:
                print(f"[HPCLibraryProfiler] Unknown operation '{op_name}', skipping.")
                continue
            bench_fn = available_ops[op_name]
            rows.extend(bench_fn())
 
        return rows
 
    def compare_libraries(
        self, operation: str = "matrix_multiply", size: int = 256
    ) -> List[Dict]:
        """
        Compare all available libraries for a single *operation* at one *size*.
 
        Returns a list sorted by composite score (descending).
        """
        rows = self.profile_all(operations=[operation])
        filtered = [r for r in rows if r.get("size") == size]
        return sorted(filtered, key=lambda r: r.get("COMPOSITE_SCORE", 0), reverse=True)
 
    def recommend_library(
        self, operation: str = "matrix_multiply", size: int = 256
    ) -> Dict:
        """
        Return the single best (library, precision) choice for the given
        operation/size pair, ranked by composite score.
        """
        results = self.compare_libraries(operation, size)
        if not results:
            return {}
        best = results[0]
        return {
            "library":         best.get("library"),
            "precision":       best.get("precision"),
            "composite_score": best.get("COMPOSITE_SCORE"),
            "score_grade":     best.get("SCORE_GRADE"),
            "elapsed_ms":      best.get("elapsed_ms"),
            "CU":  best.get("CU"),
            "EU":  best.get("EU"),
            "CO2": best.get("CO2"),
            "$":   best.get("$"),
        }
 
    # ------------------------------------------------------------------
    # Per-operation benchmarks
    # ------------------------------------------------------------------
 
    def _bench_gemm(self) -> List[Dict]:
        rows: List[Dict] = []
        for lib_name, xp, hw in self._available_backends():
            for precision in self.precisions:
                for n in self.sizes:
                    A = _make_matrix(n, precision, xp)
                    B = _make_matrix(n, precision, xp)
 
                    def _run():
                        _ = xp.dot(A, B)
                        if hw == "gpu":
                            xp.cuda.Stream.null.synchronize()
 
                    try:
                        elapsed = _benchmark(_run, self.repeats)
                    except Exception as exc:
                        elapsed = float("nan")
                        print(f"[GEMM {lib_name}/{precision}/n={n}] skipped: {exc}")
 
                    flops = _flops_gemm(n)
                    rows.append(self._make_row(
                        "matrix_multiply", lib_name, precision, n,
                        elapsed, flops, hw
                    ))
        return rows
 
    def _bench_fft(self) -> List[Dict]:
        rows: List[Dict] = []
        for lib_name, xp, hw in self._available_backends():
            for precision in self.precisions:
                for n in self.sizes:
                    x = _make_vector(n * n, precision, xp)
 
                    # Use scipy.fft for CPU, cupy.fft for GPU
                    if hw == "cpu" and sp_fft is not None and lib_name == "scipy":
                        fft_fn = sp_fft.fft
                    else:
                        fft_fn = xp.fft.fft
 
                    def _run(v=x):
                        _ = fft_fn(v)
                        if hw == "gpu":
                            xp.cuda.Stream.null.synchronize()
 
                    try:
                        elapsed = _benchmark(_run, self.repeats)
                    except Exception as exc:
                        elapsed = float("nan")
                        print(f"[FFT {lib_name}/{precision}/n={n}] skipped: {exc}")
 
                    flops = _flops_fft(n * n)
                    rows.append(self._make_row(
                        "fft", lib_name, precision, n, elapsed, flops, hw
                    ))
        return rows
 
    def _bench_linear_solve(self) -> List[Dict]:
        rows: List[Dict] = []
        for lib_name, xp, hw in self._available_backends():
            for precision in self.precisions:
                if precision in ("FP16", "INT8"):
                    continue  # LAPACK solvers require at least FP32
                for n in self.sizes:
                    A = _make_matrix(n, precision, xp)
                    # Make A diagonally dominant to ensure non-singularity
                    A = A + xp.eye(n, dtype=A.dtype) * n
                    b = _make_vector(n, precision, xp)
 
                    if hw == "cpu" and sp_linalg is not None and lib_name == "scipy":
                        def _run(a=A, bv=b):
                            _ = sp_linalg.solve(a, bv)
                    else:
                        def _run(a=A, bv=b):
                            _ = xp.linalg.solve(a, bv)
                            if hw == "gpu":
                                xp.cuda.Stream.null.synchronize()
 
                    try:
                        elapsed = _benchmark(_run, self.repeats)
                    except Exception as exc:
                        elapsed = float("nan")
                        print(f"[Solve {lib_name}/{precision}/n={n}] skipped: {exc}")
 
                    flops = _flops_linear_solve(n)
                    rows.append(self._make_row(
                        "linear_solve", lib_name, precision, n,
                        elapsed, flops, hw
                    ))
        return rows
 
    def _bench_svd(self) -> List[Dict]:
        rows: List[Dict] = []
        for lib_name, xp, hw in self._available_backends():
            for precision in self.precisions:
                if precision in ("FP16", "INT8"):
                    continue  # SVD requires FP32 or better
                for n in self.sizes:
                    if n > 256:
                        continue  # SVD is O(n³) — skip large sizes
                    A = _make_matrix(n, precision, xp)
 
                    def _run(a=A):
                        _ = xp.linalg.svd(a, full_matrices=False)
                        if hw == "gpu":
                            xp.cuda.Stream.null.synchronize()
 
                    try:
                        elapsed = _benchmark(_run, self.repeats)
                    except Exception as exc:
                        elapsed = float("nan")
                        print(f"[SVD {lib_name}/{precision}/n={n}] skipped: {exc}")
 
                    flops = _flops_svd(n)
                    rows.append(self._make_row(
                        "svd", lib_name, precision, n, elapsed, flops, hw
                    ))
        return rows
 
    def _bench_norm(self) -> List[Dict]:
        rows: List[Dict] = []
        for lib_name, xp, hw in self._available_backends():
            for precision in self.precisions:
                for n in self.sizes:
                    x = _make_vector(n * n, precision, xp)
 
                    def _run(v=x):
                        _ = xp.linalg.norm(v)
                        if hw == "gpu":
                            xp.cuda.Stream.null.synchronize()
 
                    try:
                        elapsed = _benchmark(_run, self.repeats)
                    except Exception as exc:
                        elapsed = float("nan")
                        print(f"[Norm {lib_name}/{precision}/n={n}] skipped: {exc}")
 
                    flops = _flops_norm(n * n)
                    rows.append(self._make_row(
                        "norm", lib_name, precision, n, elapsed, flops, hw
                    ))
        return rows
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def _available_backends(self) -> List[Tuple[str, Any, str]]:
        """Return list of (lib_name, xp_module, hw_tag) for installed libs."""
        backends: List[Tuple[str, Any, str]] = []
        if np is not None:
            backends.append(("numpy", np, "cpu"))
        if sp is not None:
            backends.append(("scipy", np, "cpu"))  # scipy uses numpy arrays
        if cupy is not None:
            try:
                cupy.array([1.0])  # test CUDA availability
                backends.append(("cupy", cupy, "gpu"))
            except Exception:
                pass
        return backends
 
    def _make_row(
        self,
        operation: str,
        library: str,
        precision: str,
        size: int,
        elapsed_s: float,
        flop_count: int,
        hw: str,
    ) -> Dict:
        """Build a result row with metrics, accuracy info, and composite score."""
        import math as _math
        if _math.isnan(elapsed_s):
            return {
                "operation": operation, "library": library,
                "precision": precision, "size": size,
                "elapsed_ms": float("nan"),
                "CU": float("nan"), "EU": float("nan"),
                "CO2": float("nan"), "$": float("nan"),
                "COMPOSITE_SCORE": 0.0, "SCORE_GRADE": "F",
            }
 
        raw = _elapsed_to_metrics(elapsed_s, flop_count, hw)
 
        # Attach precision-accuracy metadata
        if precision in self._prec_model.specs:
            adj = self._prec_model.adjust_metrics(raw, precision, "FP64")
            metrics = adj.adjusted_metrics
            accuracy = adj.accuracy_info
        else:
            metrics  = raw
            accuracy = {}
 
        scored = self._scorer.calculate_composite_score(metrics)
 
        row: Dict = {
            "operation":   operation,
            "library":     library,
            "precision":   precision,
            "size":        size,
            "hardware":    hw,
            "elapsed_ms":  round(elapsed_s * 1000, 4),
            "flops":       flop_count,
        }
        row.update(metrics)
        row.update(accuracy)
        row["COMPOSITE_SCORE"]  = scored.get("COMPOSITE_SCORE", 0.0)
        row["SCORE_GRADE"]      = scored.get("SCORE_GRADE", "F")
        row["EFFICIENCY_RATING"] = scored.get("EFFICIENCY_RATING", "")
        return row
 
 
# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
 
def run_hpc_comparison(
    operations: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    precisions: Optional[List[str]] = None,
    composite_profile: str = "HPC",
    verbose: bool = True,
) -> List[Dict]:
    """
    One-shot helper: profile HPC operations and print a summary table.
 
    Returns the raw list of result dicts.
    """
    profiler = HPCLibraryProfiler(
        sizes=sizes,
        precisions=precisions,
        composite_profile=composite_profile,
    )
    results = profiler.profile_all(operations=operations)
 
    if verbose:
        _print_summary(results)
 
    return results
 
 
def _print_summary(rows: List[Dict]) -> None:
    """Print a compact comparison table to stdout."""
    if not rows:
        print("No results to display.")
        return
 
    header = f"{'Operation':<20} {'Library':<8} {'Precision':<8} {'Size':>5} {'ms':>8} {'CU':>8} {'Score':>7} {'Grade':<4}"
    print("\n" + "=" * len(header))
    print("HPC Library Comparison — CCP Cost Metrics")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
 
    for r in rows:
        if isinstance(r.get("elapsed_ms"), float) and r["elapsed_ms"] != r["elapsed_ms"]:
            continue
        print(
            f"{r.get('operation',''):<20} "
            f"{r.get('library',''):<8} "
            f"{r.get('precision',''):<8} "
            f"{r.get('size',0):>5} "
            f"{r.get('elapsed_ms',0):>8.2f} "
            f"{r.get('CU',0):>8.4f} "
            f"{r.get('COMPOSITE_SCORE',0):>7.2f} "
            f"{r.get('SCORE_GRADE',''):>4}"
        )
    print("=" * len(header) + "\n")