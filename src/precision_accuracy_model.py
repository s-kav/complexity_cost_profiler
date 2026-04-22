# complexity_cost_profiler/src/precision_accuracy_model.py

"""
Precision-Accuracy Model for trade-off analysis
 
Maps floating-point precision formats (FP64, FP32, BF16, FP16, INT8) to:
  - Cost multipliers for CU, EU, CO2, $
  - Accuracy characteristics (machine epsilon, max relative error, dynamic range)
  - Tradeoff analysis: what do we gain/lose by switching precision?
 
Reference data: NVIDIA A100 SXM4 GPU specs, IEEE 754 standard.
"""
 
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
 
_HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION_PROFILES_PATH = os.path.join(_HERE, "..", "cost_models", "precision_profiles.json")
 
SUPPORTED_PRECISIONS = ["FP64", "FP32", "BF16", "FP16", "INT8"]
 
 
@dataclass
class PrecisionSpec:
    name: str
    full_name: str
    total_bits: int
    mantissa_bits: int
    exponent_bits: int
    machine_epsilon: float
    max_relative_error: float
    dynamic_range_decades: float
    throughput_multiplier_cuda: float
    throughput_multiplier_tensor: float
    cu_multiplier: float
    eu_multiplier: float
    co2_multiplier: float
    cost_multiplier: float
    typical_use: str
 
 
@dataclass
class PrecisionTradeoff:
    """Result of comparing two precision formats."""
    from_precision: str
    to_precision: str
    speedup: float
    energy_reduction_pct: float
    co2_reduction_pct: float
    cost_reduction_pct: float
    accuracy_loss_factor: float
    epsilon_from: float
    epsilon_to: float
    dynamic_range_from: float
    dynamic_range_to: float
    recommendation: str
    warnings: List[str] = field(default_factory=list)
 
 
@dataclass
class PrecisionAwareMetrics:
    """Cost metrics adjusted for a specific precision format."""
    precision: str
    base_metrics: Dict[str, float]
    adjusted_metrics: Dict[str, float]
    accuracy_info: Dict[str, float]
    composite_score: Optional[float] = None
    score_grade: Optional[str] = None
 
 
class PrecisionAccuracyModel:
    """
    Model for analyzing cost-accuracy tradeoffs across precision formats.
 
    Quick usage::
 
        model = PrecisionAccuracyModel()
 
        # Scale FP64 baseline metrics to FP16
        adj = model.adjust_metrics({"CU": 100, "EU": 1.0, "CO2": 0.1, "$": 0.01}, "FP16")
 
        # Summarise the FP64 → FP16 tradeoff
        t = model.compare_precisions("FP64", "FP16")
        print(t.speedup, t.accuracy_loss_factor)
 
        # Full table for all precisions
        rows = model.build_tradeoff_table(base_metrics)
    """
 
    def __init__(self, profiles_path: str = PRECISION_PROFILES_PATH) -> None:
        self._data = self._load_profiles(profiles_path)
        self.specs: Dict[str, PrecisionSpec] = self._build_specs()
        self.task_requirements: Dict[str, Dict] = self._data.get("task_accuracy_requirements", {})
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _load_profiles(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_profiles()
 
    def _build_specs(self) -> Dict[str, PrecisionSpec]:
        specs: Dict[str, PrecisionSpec] = {}
        for name, d in self._data["precisions"].items():
            specs[name] = PrecisionSpec(
                name=name,
                full_name=d["full_name"],
                total_bits=d["total_bits"],
                mantissa_bits=d["mantissa_bits"],
                exponent_bits=d["exponent_bits"],
                machine_epsilon=d["machine_epsilon"],
                max_relative_error=d["max_relative_error"],
                dynamic_range_decades=d["dynamic_range_decades"],
                throughput_multiplier_cuda=d["throughput_multiplier_cuda"],
                throughput_multiplier_tensor=d["throughput_multiplier_tensor"],
                cu_multiplier=d["cu_multiplier"],
                eu_multiplier=d["eu_multiplier"],
                co2_multiplier=d["co2_multiplier"],
                cost_multiplier=d["cost_multiplier"],
                typical_use=d["typical_use"],
            )
        return specs
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def adjust_metrics(
        self,
        base_metrics: Dict[str, float],
        precision: str,
        base_precision: str = "FP64",
    ) -> PrecisionAwareMetrics:
        """
        Adjust cost metrics from *base_precision* to *precision*.
 
        Args:
            base_metrics:    Original metrics dict with CU, EU, CO2, $ keys.
            precision:       Target precision format name.
            base_precision:  Precision the base_metrics were computed for.
 
        Returns:
            PrecisionAwareMetrics with scaled values and accuracy information.
        """
        if precision not in self.specs:
            raise ValueError(f"Unknown precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}")
        if base_precision not in self.specs:
            raise ValueError(f"Unknown base precision '{base_precision}'. Supported: {SUPPORTED_PRECISIONS}")
 
        spec = self.specs[precision]
        base = self.specs[base_precision]
 
        cu_f  = spec.cu_multiplier   / base.cu_multiplier
        eu_f  = spec.eu_multiplier   / base.eu_multiplier
        co2_f = spec.co2_multiplier  / base.co2_multiplier
        usd_f = spec.cost_multiplier / base.cost_multiplier
 
        adjusted = {
            "CU":  base_metrics.get("CU",  0.0) * cu_f,
            "EU":  base_metrics.get("EU",  0.0) * eu_f,
            "CO2": base_metrics.get("CO2", 0.0) * co2_f,
            "$":   base_metrics.get("$",   0.0) * usd_f,
        }
 
        accuracy_info = {
            "machine_epsilon":       spec.machine_epsilon,
            "max_relative_error":    spec.max_relative_error,
            "dynamic_range_decades": spec.dynamic_range_decades,
            "mantissa_bits":         float(spec.mantissa_bits),
            "total_bits":            float(spec.total_bits),
            "throughput_multiplier": spec.throughput_multiplier_cuda,
        }
 
        return PrecisionAwareMetrics(
            precision=precision,
            base_metrics=base_metrics.copy(),
            adjusted_metrics=adjusted,
            accuracy_info=accuracy_info,
        )
 
    def compare_precisions(
        self, from_precision: str, to_precision: str
    ) -> PrecisionTradeoff:
        """
        Compute the tradeoff when switching from *from_precision* to *to_precision*.
 
        Returns a PrecisionTradeoff dataclass with speedup, savings, accuracy
        loss factor, human-readable recommendation, and any warnings.
        """
        for p in (from_precision, to_precision):
            if p not in self.specs:
                raise ValueError(f"Unknown precision '{p}'. Supported: {SUPPORTED_PRECISIONS}")
 
        src = self.specs[from_precision]
        dst = self.specs[to_precision]
 
        speedup         = src.cu_multiplier / dst.cu_multiplier
        energy_red_pct  = (1.0 - dst.eu_multiplier  / src.eu_multiplier)  * 100
        co2_red_pct     = (1.0 - dst.co2_multiplier / src.co2_multiplier) * 100
        cost_red_pct    = (1.0 - dst.cost_multiplier / src.cost_multiplier) * 100
        acc_loss        = dst.max_relative_error / src.max_relative_error if src.max_relative_error > 0 else 1.0
 
        warnings: List[str] = []
 
        # Warn only on severe dynamic range reduction (below 10 decades)
        if dst.dynamic_range_decades < 10:
            warnings.append(
                f"Limited dynamic range: {dst.dynamic_range_decades:.0f} decades "
                f"(down from {src.dynamic_range_decades:.0f}) — overflow/underflow risk."
            )
 
        # Use absolute target max_relative_error to classify accuracy level.
        # Thresholds chosen so that:
        #   FP32 (5.96e-8): no warning — widely acceptable for scientific HPC
        #   FP16 (4.88e-4): moderate warning — needs loss scaling in DL
        #   BF16 (3.91e-3): training warning — fine for DL, not for science
        #   INT8 (~4e-3 + integer): quantization-only warning
        target_err = dst.max_relative_error
        if to_precision == "INT8":
            warnings.append(
                f"Integer format — max relative error ≈{target_err:.1e}. "
                "Requires explicit quantization (calibration, zero-point, scale)."
            )
        elif target_err >= 1e-3:
            warnings.append(
                f"Low precision (ε≈{target_err:.1e}): suitable for DL training with "
                "gradient averaging, but not for iterative scientific HPC kernels."
            )
        elif target_err >= 1e-4:
            warnings.append(
                f"Moderate precision (ε≈{target_err:.1e}) — validate with loss scaling "
                "or mixed-precision accumulation before production use."
            )
            
        if to_precision == "INT8" and from_precision in ("FP64", "FP32"):
            warnings.append(
                "INT8 requires explicit quantization (calibration, zero-point, scale). "
                "Cannot be used as a drop-in replacement."
            )
 
        recommendation = self._build_recommendation(
            from_precision, to_precision, speedup, energy_red_pct, acc_loss, warnings
        )
 
        return PrecisionTradeoff(
            from_precision=from_precision,
            to_precision=to_precision,
            speedup=speedup,
            energy_reduction_pct=energy_red_pct,
            co2_reduction_pct=co2_red_pct,
            cost_reduction_pct=cost_red_pct,
            accuracy_loss_factor=acc_loss,
            epsilon_from=src.machine_epsilon,
            epsilon_to=dst.machine_epsilon,
            dynamic_range_from=src.dynamic_range_decades,
            dynamic_range_to=dst.dynamic_range_decades,
            recommendation=recommendation,
            warnings=warnings,
        )
 
    def recommend_precision(
        self,
        base_metrics: Dict[str, float],
        task_type: str = "general_scientific",
    ) -> Dict[str, PrecisionAwareMetrics]:
        """
        Return all precision-adjusted metrics for candidates that satisfy
        the accuracy requirements of *task_type*, ordered from highest to
        lowest precision.
 
        Args:
            base_metrics: Baseline cost metrics computed assuming FP64.
            task_type:    Key from ``task_accuracy_requirements`` in the JSON.
 
        Returns:
            Ordered dict mapping precision name → PrecisionAwareMetrics.
        """
        req = self.task_requirements.get(task_type, {})
        min_precision = req.get("min_precision", "FP64")
        min_idx = SUPPORTED_PRECISIONS.index(min_precision) if min_precision in SUPPORTED_PRECISIONS else 0
        candidates = SUPPORTED_PRECISIONS[min_idx:]
 
        return {p: self.adjust_metrics(base_metrics, p, "FP64") for p in candidates if p in self.specs}
 
    def build_tradeoff_table(
        self,
        base_metrics: Dict[str, float],
        base_precision: str = "FP64",
    ) -> List[Dict]:
        """
        Build a comparison table (list of dicts) for all precisions.
 
        Each row contains precision specs, speedup, savings percentages,
        accuracy loss factor, and the adjusted cost metrics.
        Suitable for direct use as a pandas DataFrame.
        """
        base_spec = self.specs[base_precision]
        rows: List[Dict] = []
 
        for precision in SUPPORTED_PRECISIONS:
            if precision not in self.specs:
                continue
            spec = self.specs[precision]
            adj  = self.adjust_metrics(base_metrics, precision, base_precision)
 
            speedup        = base_spec.cu_multiplier / spec.cu_multiplier
            energy_saved   = (1.0 - spec.eu_multiplier  / base_spec.eu_multiplier)  * 100
            co2_saved      = (1.0 - spec.co2_multiplier / base_spec.co2_multiplier) * 100
            cost_saved     = (1.0 - spec.cost_multiplier / base_spec.cost_multiplier) * 100
            acc_loss       = spec.max_relative_error / base_spec.max_relative_error
 
            rows.append({
                "Precision":            precision,
                "Bits":                 spec.total_bits,
                "Machine_epsilon":      spec.machine_epsilon,
                "Max_rel_error":        spec.max_relative_error,
                "Dynamic_range_dec":    spec.dynamic_range_decades,
                "Speedup_vs_base":      round(speedup, 2),
                "Energy_saved_%":       round(energy_saved, 1),
                "CO2_saved_%":          round(co2_saved, 1),
                "Cost_saved_%":         round(cost_saved, 1),
                "Accuracy_loss_factor": round(acc_loss, 1) if acc_loss >= 1.0 else round(acc_loss, 6),
                "CU":                   round(adj.adjusted_metrics["CU"],  4),
                "EU":                   round(adj.adjusted_metrics["EU"],  8),
                "CO2":                  round(adj.adjusted_metrics["CO2"], 9),
                "$":                    round(adj.adjusted_metrics["$"],   9),
                "Typical_use":          spec.typical_use,
            })
 
        return rows
 
    def get_spec(self, precision: str) -> PrecisionSpec:
        """Return the PrecisionSpec for *precision*."""
        if precision not in self.specs:
            raise ValueError(f"Unknown precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}")
        return self.specs[precision]
 
    def list_precisions(self) -> List[str]:
        """Return list of all supported precision names."""
        return [p for p in SUPPORTED_PRECISIONS if p in self.specs]
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _build_recommendation(
        from_p: str, to_p: str,
        speedup: float, energy_red_pct: float,
        acc_loss: float, warnings: List[str],
    ) -> str:
        
        is_int8_warning     = any("integer format" in w.lower() or
                                     ("quantiz" in w.lower() and "explicit" in w.lower())
                                     for w in warnings)
        is_low_prec_warning = any("low precision" in w.lower() for w in warnings)
        is_mod_prec_warning = any("moderate precision" in w.lower() for w in warnings)
        is_overflow_warning = any("dynamic range" in w.lower() for w in warnings)
 
        if not warnings:
            return (
                f"Recommended: {to_p} offers {speedup:.1f}x speedup and "
                f"{energy_red_pct:.0f}% energy reduction with acceptable accuracy."
            )
        if is_int8_warning:
            return (
                f"Inference-only: {to_p} is {speedup:.1f}x faster but requires "
                "explicit quantization — not a drop-in replacement for FP compute."
            )
        if is_low_prec_warning and is_overflow_warning:
            return (
                f"DL-training only: {to_p} gives {speedup:.1f}x speedup but limited "
                "dynamic range and low precision restrict use to deep learning training."
            )
        if is_low_prec_warning:
            return (
                f"DL-friendly: {to_p} gives {speedup:.1f}x speedup — suitable for "
                "deep learning training (gradient averaging compensates low precision), "
                "not for iterative scientific solvers."
            )
        if is_mod_prec_warning:
            return (
                f"Conditional: {to_p} gives {speedup:.1f}x speedup but requires "
                "loss scaling or validation for your specific workload."
            )
        if is_overflow_warning:
            return (
                f"Caution: {to_p} gives {speedup:.1f}x speedup but limited dynamic "
                "range may cause overflow — monitor for Inf/NaN."
            f"{to_p} provides {speedup:.1f}x speedup with {energy_red_pct:.0f}% "
            "energy savings. Validate numerical accuracy for your specific algorithm."
        )
 
    @staticmethod
    def _default_profiles() -> dict:
        """Minimal built-in fallback when the JSON file is unavailable."""
        return {
            "precisions": {
                "FP64": {
                    "full_name": "Double Precision Float", "total_bits": 64,
                    "mantissa_bits": 52, "exponent_bits": 11,
                    "machine_epsilon": 2.22e-16, "max_relative_error": 1.11e-16,
                    "dynamic_range_decades": 307,
                    "throughput_multiplier_cuda": 1.0, "throughput_multiplier_tensor": 2.0,
                    "cu_multiplier": 1.0, "eu_multiplier": 1.0,
                    "co2_multiplier": 1.0, "cost_multiplier": 1.0,
                    "typical_use": "Scientific computing, CFD, climate, finance",
                },
                "FP32": {
                    "full_name": "Single Precision Float", "total_bits": 32,
                    "mantissa_bits": 23, "exponent_bits": 8,
                    "machine_epsilon": 1.19e-7, "max_relative_error": 5.96e-8,
                    "dynamic_range_decades": 38,
                    "throughput_multiplier_cuda": 2.0, "throughput_multiplier_tensor": 16.0,
                    "cu_multiplier": 0.5, "eu_multiplier": 0.55,
                    "co2_multiplier": 0.55, "cost_multiplier": 0.55,
                    "typical_use": "General HPC, deep learning, graphics",
                },
                "BF16": {
                    "full_name": "Brain Float 16", "total_bits": 16,
                    "mantissa_bits": 7, "exponent_bits": 8,
                    "machine_epsilon": 7.81e-3, "max_relative_error": 3.91e-3,
                    "dynamic_range_decades": 38,
                    "throughput_multiplier_cuda": 32.0, "throughput_multiplier_tensor": 32.0,
                    "cu_multiplier": 0.125, "eu_multiplier": 0.20,
                    "co2_multiplier": 0.20, "cost_multiplier": 0.20,
                    "typical_use": "DL training — preferred over FP16 for stability",
                },
                "FP16": {
                    "full_name": "Half Precision Float", "total_bits": 16,
                    "mantissa_bits": 10, "exponent_bits": 5,
                    "machine_epsilon": 9.77e-4, "max_relative_error": 4.88e-4,
                    "dynamic_range_decades": 4,
                    "throughput_multiplier_cuda": 32.0, "throughput_multiplier_tensor": 32.0,
                    "cu_multiplier": 0.125, "eu_multiplier": 0.18,
                    "co2_multiplier": 0.18, "cost_multiplier": 0.18,
                    "typical_use": "NN training/inference, mixed-precision ML",
                },
                "INT8": {
                    "full_name": "8-bit Integer", "total_bits": 8,
                    "mantissa_bits": 0, "exponent_bits": 0,
                    "machine_epsilon": 1.0, "max_relative_error": 0.004,
                    "dynamic_range_decades": 2.3,
                    "throughput_multiplier_cuda": 64.0, "throughput_multiplier_tensor": 64.0,
                    "cu_multiplier": 0.0625, "eu_multiplier": 0.10,
                    "co2_multiplier": 0.10, "cost_multiplier": 0.10,
                    "typical_use": "Inference-only, post-training quantization",
                },
            },
            "task_accuracy_requirements": {},
        }