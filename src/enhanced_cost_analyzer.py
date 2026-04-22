# complexity_cost_profiler/src/enhanced_cost_analyzer.py

import os
import dis
import copy
import inspect
import importlib.util
import statistics
import requests
from typing import Dict, Tuple, Callable, Any, Optional, List
from instruction_cost_model import InstructionCostModel
from composite_score_calculator import CompositeScoreCalculator, PROFILE_WEIGHTS, DEFAULT_COMPOSITE_WEIGHTS
from precision_accuracy_model import PrecisionAccuracyModel, SUPPORTED_PRECISIONS

# Import from our other project modules

# Constants used specifically by the analyzer
CARBON_INTENSITY_API = "https://api.carbonintensity.org.uk/intensity"
DEFAULT_CARBON_INTENSITY = 0.2  # Fallback value in kgCO2/kWh


class EnhancedCostAnalyzer:
    """
    Enhanced analyzer class for evaluating algorithm costs with composite scoring.

    Provides methods to analyze Python functions, LLVM IR, and PTX code
    to estimate computational costs across multiple metrics, including
    unified composite scores for algorithm comparison.
    """

    def __init__(self, arch: str = "x86_64",
                 composite_weights: Optional[Dict[str, float]] = None,
                 profile: str = "DEFAULT") -> None:
        """
        Initialize enhanced cost analyzer with specified architecture.

        Args:
            arch: Target architecture for cost model
            composite_weights: Custom weights for composite score calculation
            profile: Predefined profile name ('HPC', 'MOBILE', 'COMMERCIAL', 'RESEARCH', 'DEFAULT')
        """
        self.arch = arch
        self.profile = profile
        self.weights = copy.deepcopy(composite_weights) if composite_weights \
                       else copy.deepcopy(PROFILE_WEIGHTS.get(profile, DEFAULT_COMPOSITE_WEIGHTS))
        self.model = InstructionCostModel(arch=arch)
        self.composite_calculator = CompositeScoreCalculator(
            weights=self.weights,
            profile=profile
        )
        self.benchmark_history = []

    def analyze_function(self, fn: Callable[..., Any],
                        include_composite: bool = True) -> Dict[str, float]:
        """
        Analyze Python function bytecode and calculate costs with composite score.
        """
        instructions = list(dis.get_instructions(fn))
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}

        for instr in instructions:
            cu, eu, co2, money = self.model.get_cost(instr.opname, is_bytecode=True)
            summary["CU"] += cu
            summary["EU"] += eu
            summary["CO2"] += co2
            summary["$"] += money

        if include_composite:
            summary = self.composite_calculator.calculate_composite_score(summary)

        return summary

    def analyze_llvm_ir(self, ir_path: str, include_composite: bool = True) -> Dict[str, float]:
        """
        Analyze LLVM IR file and calculate costs with composite score.
        """
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}

        with open(ir_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'): continue
                tokens = line.split()
                if tokens:
                    opcode = None
                    for token in tokens:
                        if '=' not in token and not token.startswith('%'):
                            opcode = token.upper()
                            break
                    if opcode:
                        cu, eu, co2, money = self.model.get_cost(opcode)
                        summary["CU"] += cu
                        summary["EU"] += eu
                        summary["CO2"] += co2
                        summary["$"] += money

        if include_composite:
            summary = self.composite_calculator.calculate_composite_score(summary)
        return summary

    def analyze_ptx(self, ptx_path: str, include_composite: bool = True) -> Dict[str, float]:
        """
        Analyze PTX (Parallel Thread Execution) file and calculate costs with composite score.
        """
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}

        with open(ptx_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith((';', '//', '.', '{')): continue
                tokens = line.split()
                if tokens:
                    instr = tokens[0].upper().rstrip(':')
                    if not instr.endswith(':') and not instr.startswith('.'):
                        cu, eu, co2, money = self.model.get_cost(instr)
                        summary["CU"] += cu
                        summary["EU"] += eu
                        summary["CO2"] += co2
                        summary["$"] += money

        if include_composite:
            summary = self.composite_calculator.calculate_composite_score(summary)
        return summary

    def fetch_carbon_intensity(self) -> float:
        """
        Fetch current carbon intensity from external API.
        """
        try:
            response = requests.get(CARBON_INTENSITY_API, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["intensity"]["actual"] / 1000.0
        except (requests.RequestException, KeyError, IndexError):
            pass
        return DEFAULT_CARBON_INTENSITY

    def compare_functions(self, fn_old: Callable[..., Any], fn_new: Callable[..., Any], include_composite: bool = True) -> Dict[str, Any]:
        """
        Compare costs between two functions (differential analysis) with composite scores.
        """
        old_cost = self.analyze_function(fn_old, include_composite)
        new_cost = self.analyze_function(fn_new, include_composite)
        differences = {}
        for key in ["CU", "EU", "CO2", "$"]:
            differences[f"{key}_diff"] = new_cost[key] - old_cost[key]
            differences[f"{key}_ratio"] = (new_cost[key] / old_cost[key]) if old_cost[key] > 0 else float('inf')
            differences[f"{key}_percent_change"] = (((new_cost[key] - old_cost[key]) / old_cost[key]) * 100 if old_cost[key] > 0 else float('inf'))
        if include_composite:
            differences["COMPOSITE_SCORE_diff"] = new_cost["COMPOSITE_SCORE"] - old_cost["COMPOSITE_SCORE"]
            differences["improvement"] = new_cost["COMPOSITE_SCORE"] > old_cost["COMPOSITE_SCORE"]
            differences["old_grade"] = old_cost["SCORE_GRADE"]
            differences["new_grade"] = new_cost["SCORE_GRADE"]
        return {"old_metrics": old_cost, "new_metrics": new_cost, "comparison": differences}

    def benchmark_suite(self, functions: List[Tuple[str, Callable[..., Any]]]) -> Dict[str, Any]:
        """
        Run benchmark suite, calibrate reference values, and then recalculate scores.
        """
        raw_results = {name: self.analyze_function(func, include_composite=False) for name, func in functions}
        raw_metrics_list = list(raw_results.values())
        self.composite_calculator.update_reference_values(raw_metrics_list)
        self.benchmark_history.extend(raw_metrics_list)
        final_results = {name: self.composite_calculator.calculate_composite_score(raw_result) for name, raw_result in raw_results.items()}
        stats = self._calculate_benchmark_stats(final_results)
        return {"results": final_results, "statistics": stats, "updated_references": self.composite_calculator.reference_values}

    def _calculate_benchmark_stats(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate statistical summary of benchmark results."""
        composite_scores = [result["COMPOSITE_SCORE"] for result in results.values()]
        return {
            "best_algorithm": max(results, key=lambda k: results[k]["COMPOSITE_SCORE"]),
            "worst_algorithm": min(results, key=lambda k: results[k]["COMPOSITE_SCORE"]),
            "average_composite_score": statistics.mean(composite_scores),
            "median_composite_score": statistics.median(composite_scores),
            "composite_score_std": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
            "score_range": max(composite_scores) - min(composite_scores)
        }
    
    def analyze_py_file(self, file_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Dynamically imports a Python file and analyzes all top-level functions and methods.
        """
        all_executable_results = []
        try:
            module_name = f"dynamic_module_{os.path.basename(file_path).replace('.py', '')}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                if verbose: print(f"Warning: Could not create module spec for {file_path}")
                return []
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            functions_to_analyze = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module_name:
                    functions_to_analyze.append((name, obj))
                elif inspect.isclass(obj) and obj.__module__ == module_name:
                    for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                        functions_to_analyze.append((f"{name}.{method_name}", method_obj))

            if not functions_to_analyze:
                return [{'Source File': os.path.basename(file_path), 'Function Name': '[No functions or methods found]'}]

            for name, executable_obj in functions_to_analyze:
                if verbose: print(f"    > Found '{name}' in {os.path.basename(file_path)}, analyzing...")
                result = self.analyze_function(executable_obj)
                result.update({'Source File': os.path.basename(file_path), 'Function Name': name})
                all_executable_results.append(result)

        except Exception as e:
            if verbose: print(f"Error analyzing Python file {file_path}: {e}")
            return [{'Source File': os.path.basename(file_path), 'Function Name': f'[Error: {e}]'}]
            
        return all_executable_results

# ------------------------------------------------------------------
# Precision-aware analysis
# ------------------------------------------------------------------

    def analyze_function_with_precision(
        self,
        fn: Callable[..., Any],
        precision: str,
        include_composite: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyse *fn* and scale the resulting metrics to *precision*.

        The bytecode analysis is always done in the same way; the precision
        model then adjusts CU/EU/CO2/$ according to the target hardware
        throughput ratios (NVIDIA A100 baseline).

        Args:
            fn:                Python callable to analyse.
            precision:         Target precision: 'FP64', 'FP32', 'BF16',
                               'FP16', or 'INT8'.
            include_composite: Whether to append composite score fields.

        Returns:
            Dict with adjusted metrics plus ``PRECISION``, ``machine_epsilon``,
            ``max_relative_error``, and ``dynamic_range_decades`` keys.
        """
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision}'. Supported: {SUPPORTED_PRECISIONS}"
            )

        base = self.analyze_function(fn, include_composite=False)
        prec_model = PrecisionAccuracyModel()
        adj = prec_model.adjust_metrics(base, precision, base_precision="FP64")

        result = adj.adjusted_metrics.copy()
        result["PRECISION"] = precision
        result.update(adj.accuracy_info)

        if include_composite:
            result = self.composite_calculator.calculate_composite_score(result)

        return result

    def compare_precisions(
        self,
        fn: Callable[..., Any],
        precisions: Optional[List[str]] = None,
        include_composite: bool = True,
    ) -> Dict[str, Any]:
        """
        Run *fn* analysis once, then project costs to every precision in
        *precisions* and return a side-by-side comparison.

        Args:
            fn:                Python callable to analyse.
            precisions:        List of precision names to compare.
                               Defaults to all supported precisions.
            include_composite: Whether to include composite score columns.

        Returns:
            Dict with keys:
              ``"base_metrics"``   — raw FP64-equivalent metrics,
              ``"by_precision"``   — per-precision adjusted metrics,
              ``"tradeoff_table"`` — list of comparison-row dicts,
              ``"best_for_performance"`` — precision with lowest CU,
              ``"best_for_energy"``      — precision with lowest EU,
              ``"best_balanced"``        — precision with best composite
                                           score (if include_composite).
        """
        if precisions is None:
            precisions = SUPPORTED_PRECISIONS

        base = self.analyze_function(fn, include_composite=False)
        prec_model = PrecisionAccuracyModel()
        table = prec_model.build_tradeoff_table(base, base_precision="FP64")

        by_precision: Dict[str, Dict] = {}
        for precision in precisions:
            if precision not in SUPPORTED_PRECISIONS:
                continue
            result = self.analyze_function_with_precision(fn, precision, include_composite)
            by_precision[precision] = result

        best_perf = min(by_precision, key=lambda p: by_precision[p].get("CU", float("inf")))
        best_energy = min(by_precision, key=lambda p: by_precision[p].get("EU", float("inf")))
        best_balanced = None
        if include_composite:
            best_balanced = max(
                by_precision,
                key=lambda p: by_precision[p].get("COMPOSITE_SCORE", 0.0),
            )

        return {
            "base_metrics":          base,
            "by_precision":          by_precision,
            "tradeoff_table":        table,
            "best_for_performance":  best_perf,
            "best_for_energy":       best_energy,
            "best_balanced":         best_balanced,
        }

    def analyze_ptx_with_precision(
        self,
        ptx_path: str,
        precision: Optional[str] = None,
        include_composite: bool = True,
    ) -> Dict[str, float]:
        """
        Analyse a PTX file with optional explicit precision override.

        If *precision* is None the analyzer auto-detects precision from
        instruction name suffixes (e.g. ``MUL.F16``).  An explicit value
        overrides the auto-detection for instructions without a suffix.

        Args:
            ptx_path:          Path to the PTX source file.
            precision:         Optional explicit precision ('FP64' … 'INT8').
            include_composite: Whether to append composite score fields.

        Returns:
            Standard metrics dict, optionally with composite score fields.
        """
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}

        with open(ptx_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith((";", "//", ".", "{")):
                    continue
                tokens = line.split()
                if tokens:
                    instr = tokens[0].upper().rstrip(":")
                    if not instr.endswith(":") and not instr.startswith("."):
                        cu, eu, co2, money = self.model.get_cost(
                            instr, precision=precision
                        )
                        summary["CU"]  += cu
                        summary["EU"]  += eu
                        summary["CO2"] += co2
                        summary["$"]   += money

        if precision:
            summary["PRECISION"] = precision

        if include_composite:
            summary = self.composite_calculator.calculate_composite_score(summary)

        return summary