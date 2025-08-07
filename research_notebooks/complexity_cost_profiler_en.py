"""
CostComplexityProfiler: Advanced Algorithmic Complexity Assessment Model with Unified Composite Score
----------------------------------------------------------------------------------------------------

Purpose:
This enhanced prototype implements a Python utility for evaluating computational "cost"
of algorithms considering processor architectural features (CPU/ARM/GPU),
energy consumption metrics and the ability to translate costs into monetary equivalent.
Now includes a unified composite score based on scaled normalization of 4 main metrics.

Enhanced Features:
- Composite Score calculation with configurable weights
- Multi-dimensional scaling and normalization
- Percentile-based scoring system
- Advanced statistical analysis of cost components

Formulas:
----------
Let:
- op_i: number of instructions of type i
- w_i: weight (cost) of instruction i in CU
- f_e(i): energy consumption of instruction i (in Joules)
- f_c(i): CO2 footprint (kg CO2)
- f_d(i): monetary cost of executing instruction i ($)

Then:
  COST_total = ∑(op_i × w_i)       [in CU]
  ENERGY_total = ∑(op_i × f_e(i))  [in Joules]
  CO2_total = ∑(op_i × f_c(i))     [in kg CO2]
  MONEY_total = ∑(op_i × f_d(i))   [in $ or €]

Composite Score Formula:
  Let S_cu, S_eu, S_co2, S_$ be normalized scores (0-100) for each metric
  Then: COMPOSITE_SCORE = α×S_cu + β×S_eu + γ×S_co2 + δ×S_$
  Where: α + β + γ + δ = 1 (configurable weights)
"""

import csv
import dis
import json
import math
import os
import platform
import statistics
import subprocess
from typing import Dict, Tuple, Callable, Any, Optional, List
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import requests


# Constants
CARBON_INTENSITY_API = "https://api.carbonintensity.org.uk/intensity"
DEFAULT_CARBON_INTENSITY = 0.2  # Fallback value in kgCO2/kWh
DEFAULT_COST_MODEL_PATH = "./cost_models"

# Default weights for composite score calculation
DEFAULT_COMPOSITE_WEIGHTS = {
    "CU": 0.35,    # Computational cost weight
    "EU": 0.25,    # Energy weight  
    "CO2": 0.25,   # Environmental weight
    "$": 0.15      # Monetary weight
}
RESEARCH_WEIGHTS = {
    "CU": 0.40, # Performance is the most important
    "EU": 0.30, # Energy is important
    "CO2": 0.25, # Environment is important
    "$": 0.05 # Cost is minimal
}
COMMERCIAL_WEIGHTS = {
    "CU": 0.30, # Performance is important
    "EU": 0.20, # Energy is a medium priority
    "CO2": 0.20, # Environment is a medium priority
    "$": 0.30 # Cost is a high priority
}
MOBILE_WEIGHTS = {
    "CU": 0.25, # Performance is important, but not critical
    "EU": 0.50, # Energy is the top priority
    "CO2": 0.15, # Environment is a medium priority
    "$": 0.10 # Cost is a low priority
}
HPC_WEIGHTS = {
    "CU": 0.50, # Productivity is the most important
    "EU": 0.30, # Energy is the second priority
    "CO2": 0.15, # Environment is important, but not critical
    "$": 0.05 # Cost is less important
}
PROFILE_WEIGHTS = MappingProxyType({
    "RESEARCH": RESEARCH_WEIGHTS,
    "COMMERCIAL": COMMERCIAL_WEIGHTS,
    "MOBILE": MOBILE_WEIGHTS,
    "HPC": HPC_WEIGHTS,
    "DEFAULT": DEFAULT_COMPOSITE_WEIGHTS,
})
# Reference values for normalization (can be calibrated based on benchmark suite)
REFERENCE_VALUES = {
    "CU": {"min": 1.0, "max": 1000.0, "typical": 100.0},
    "EU": {"min": 0.0001, "max": 10.0, "typical": 1.0},
    "CO2": {"min": 0.00001, "max": 1.0, "typical": 0.1},
    "$": {"min": 0.000001, "max": 0.1, "typical": 0.01}
}


class CompositeScoreCalculator:
    """
    Calculator for unified composite scores based on multiple cost metrics.
    
    Provides normalization, scaling, and weighted combination of cost metrics
    into a single composite score for algorithm comparison.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, 
                 profile: Optional[str] = None,
                 reference_values: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize composite score calculator.
        
        Args:
            weights: Weight distribution for metrics (must sum to 1.0)
            profile: Predefined profile name ('HPC', 'MOBILE', 'COMMERCIAL', 'RESEARCH', 'DEFAULT')
            reference_values: Reference values for normalization
        """
        if weights is not None:
            self.weights = weights.copy()
        elif profile is not None:
            if profile not in PROFILE_WEIGHTS:
                raise ValueError(f"Unknown profile: {profile}. Available: {list(PROFILE_WEIGHTS.keys())}")
            self.weights = PROFILE_WEIGHTS[profile].copy()
        else:
            self.weights = DEFAULT_COMPOSITE_WEIGHTS.copy()
        
        self.profile = profile or "CUSTOM"
        self.reference_values = reference_values or REFERENCE_VALUES.copy()
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def normalize_metric(self, value: float, metric: str, method: str = "minmax") -> float:
        """
        Normalize a metric value to 0-100 scale.
        
        Args:
            value: Raw metric value
            metric: Metric name (CU, EU, CO2, $)
            method: Normalization method ('minmax', 'zscore', 'log')
            
        Returns:
            Normalized score (0-100)
        """
        if metric not in self.reference_values:
            return 50.0  # Default middle value for unknown metrics
        
        ref = self.reference_values[metric]
        min_val, max_val = ref["min"], ref["max"]
        
        if method == "minmax":
            # Min-max normalization with inversion (lower is better)
            if max_val <= min_val:
                return 50.0
            normalized = (value - min_val) / (max_val - min_val)
            # Invert: lower cost = higher score
            return max(0.0, min(100.0, 100.0 * (1.0 - normalized)))
        
        elif method == "zscore":
            # Z-score normalization using typical value as mean
            typical = ref["typical"]
            std_estimate = (max_val - min_val) / 6  # Rough 6-sigma estimate
            if std_estimate <= 0:
                return 50.0
            z_score = (value - typical) / std_estimate
            # Convert to percentile and invert
            percentile = self._z_to_percentile(-z_score)  # Negative for inversion
            return max(0.0, min(100.0, percentile))
        
        elif method == "log":
            # Logarithmic normalization for highly skewed data
            if value <= 0 or min_val <= 0 or max_val <= min_val:
                return 50.0
            log_val = math.log(value)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            normalized = (log_val - log_min) / (log_max - log_min)
            return max(0.0, min(100.0, 100.0 * (1.0 - normalized)))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile using approximation."""
        # Approximation of cumulative normal distribution
        return 50.0 * (1.0 + math.erf(z_score / math.sqrt(2)))
    
    def calculate_composite_score(self, metrics: Dict[str, float], 
                                method: str = "minmax") -> Dict[str, float]:
        """
        Calculate composite score from individual metrics.
        
        Args:
            metrics: Dictionary with raw metric values
            method: Normalization method to use
            
        Returns:
            Dictionary with normalized scores and composite score
        """
        normalized_scores = {}
        
        # Normalize each metric
        for metric, value in metrics.items():
            if metric in self.weights:
                normalized_scores[f"{metric}_normalized"] = self.normalize_metric(
                    value, metric, method
                )
        
        # Calculate weighted composite score
        composite_score = 0.0
        for metric, weight in self.weights.items():
            normalized_key = f"{metric}_normalized"
            if normalized_key in normalized_scores:
                composite_score += weight * normalized_scores[normalized_key]
        
        # Add composite score to results
        result = metrics.copy()
        result.update(normalized_scores)
        result["COMPOSITE_SCORE"] = composite_score
        
        # Add score interpretation
        result["SCORE_GRADE"] = self._get_score_grade(composite_score)
        result["EFFICIENCY_RATING"] = self._get_efficiency_rating(composite_score)
        
        return result
    
    def _get_score_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "A-"
        elif score >= 75: return "B+"
        elif score >= 70: return "B"
        elif score >= 65: return "B-"
        elif score >= 60: return "C+"
        elif score >= 55: return "C"
        elif score >= 50: return "C-"
        elif score >= 40: return "D"
        else: return "F"
    
    def _get_efficiency_rating(self, score: float) -> str:
        """Convert numeric score to efficiency rating."""
        if score >= 85: return "Excellent"
        elif score >= 70: return "Good"
        elif score >= 55: return "Average"
        elif score >= 40: return "Below Average"
        else: return "Poor"
    
    def update_reference_values(self, benchmark_results: List[Dict[str, float]]) -> None:
        """
        Update reference values based on benchmark results.
        
        Args:
            benchmark_results: List of metric dictionaries from benchmark runs
        """
        if not benchmark_results:
            return
        
        for metric in self.weights.keys():
            values = [result.get(metric, 0) for result in benchmark_results if metric in result]
            if values:
                self.reference_values[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "typical": statistics.median(values)
                }

    def get_profile_info(self) -> Dict[str, Any]:
        """
        Get information about current weight profile.
        
        Returns:
            Dictionary with profile information
        """
        return {
            "profile": self.profile,
            "weights": self.weights.copy(),
            "description": self._get_profile_description()
        }

    def _get_profile_description(self) -> str:
        """Get description of current profile."""
        descriptions = {
            "HPC": "High Performance Computing - optimized for maximum computational throughput",
            "MOBILE": "Mobile/IoT - optimized for energy efficiency and battery life",
            "COMMERCIAL": "Commercial Cloud - balanced approach with cost consideration",
            "RESEARCH": "Research/Academic - focused on performance with environmental awareness",
            "DEFAULT": "Default balanced profile for general use cases",
            "CUSTOM": "Custom weight configuration"
        }
        return descriptions.get(self.profile, "Custom profile configuration")


class InstructionCostModel:
    """
    Model for instruction costs across different architectures.
    
    Loads cost models from JSON files and provides cost lookup functionality
    for different instruction types across various metrics (CU, EU, CO2, $).
    """
    
    def __init__(self, arch: str) -> None:
        """
        Initialize instruction cost model for specified architecture.
        
        Args:
            arch: Target architecture (e.g., 'x86', 'arm', 'gpu')
            
        Raises:
            RuntimeError: If cost model file for architecture is not found
        """
        self.arch = arch.lower()
        self.weights = self._load_weights()
        self.bytecode_mapping = self._get_bytecode_mapping()

    def _load_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Load instruction weights from architecture-specific JSON file.
        
        Returns:
            Dictionary mapping instruction names to cost metrics
            
        Raises:
            RuntimeError: If cost model file cannot be loaded
        """
        # Try multiple ways to find the correct path
        possible_paths = []
        
        # Method 1: Use __file__ if available
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths.append(os.path.join(script_dir, f"cost_models/{self.arch}_instr_costs.json"))
        except NameError:
            pass
        
        # Method 2: Current working directory
        possible_paths.append(f"cost_models/{self.arch}_instr_costs.json")
        
        # Method 3: Relative to current directory
        possible_paths.append(f"./{self.arch}_instr_costs.json")
        
        # Method 4: Check if DEFAULT_COST_MODEL_PATH exists
        if os.path.exists(DEFAULT_COST_MODEL_PATH):
            possible_paths.append(
                os.path.join(DEFAULT_COST_MODEL_PATH, f"{self.arch}_instr_costs.json")
            )
        
        # Try each path until one works
        for model_file in possible_paths:
            try:
                if os.path.exists(model_file):
                    with open(model_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                continue
        
        # If no file found, use default fallback model
        print(f"Warning: Cost model not found for architecture: {self.arch}. Using default values.")
        return self._get_default_cost_model()

    def _get_default_cost_model(self) -> Dict[str, Dict[str, float]]:
        """Get default cost model when no file is available."""
        return {
            "ADD": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001},
            "SUB": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001},
            "MUL": {"CU": 3.0, "EU": 0.0003, "CO2": 0.00015, "$": 0.00003},
            "DIV": {"CU": 10.0, "EU": 0.001, "CO2": 0.0005, "$": 0.0001},
            "LOAD": {"CU": 2.0, "EU": 0.0002, "CO2": 0.0001, "$": 0.00002},
            "STORE": {"CU": 2.0, "EU": 0.0002, "CO2": 0.0001, "$": 0.00002},
            "JMP": {"CU": 1.5, "EU": 0.00015, "CO2": 0.000075, "$": 0.000015},
            "CALL": {"CU": 5.0, "EU": 0.0005, "CO2": 0.00025, "$": 0.00005},
            "MOV": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001},
            "AND": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001},
            "OR": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001},
            "XOR": {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001}
        }

    def _get_bytecode_mapping(self) -> Dict[str, str]:
        """
        Get mapping from Python bytecode instructions to architecture instructions.
        
        Returns:
            Dictionary mapping bytecode ops to architecture ops
        """
        return {
            'BINARY_ADD': 'ADD',
            'BINARY_SUBTRACT': 'SUB', 
            'BINARY_MULTIPLY': 'MUL',
            'BINARY_TRUE_DIVIDE': 'DIV',
            'BINARY_FLOOR_DIVIDE': 'DIV',
            'BINARY_AND': 'AND',
            'BINARY_OR': 'OR',
            'BINARY_XOR': 'XOR',
            'LOAD_CONST': 'LOAD',
            'LOAD_FAST': 'LOAD',
            'LOAD_GLOBAL': 'LOAD',
            'STORE_FAST': 'STORE',
            'STORE_GLOBAL': 'STORE',
            'STORE_NAME': 'STORE',
            'JUMP_FORWARD': 'JMP',
            'JUMP_IF_TRUE_OR_POP': 'JMP',
            'JUMP_IF_FALSE_OR_POP': 'JMP',
            'CALL_FUNCTION': 'CALL',
            'RETURN_VALUE': 'MOV'
        }

    def get_cost(self, opname: str, is_bytecode: bool = False) -> Tuple[float, float, float, float]:
        """
        Get cost metrics for specified instruction.
        
        Args:
            opname: Instruction operation name
            is_bytecode: Whether the opname is Python bytecode instruction
            
        Returns:
            Tuple of (CU, EU, CO2, $) cost values
        """
        opname = opname.upper()
        
        # Map Python bytecode to architecture instruction if needed
        if is_bytecode and opname in self.bytecode_mapping:
            opname = self.bytecode_mapping[opname]
        
        data = self.weights.get(opname)
        
        if data is None:
            # Default fallback values for unknown instructions
            default_values = {
                "CU": 1.0, 
                "EU": 0.0001, 
                "CO2": 0.00005, 
                "$": 0.00001
            }
            return (
                default_values["CU"], 
                default_values["EU"], 
                default_values["CO2"], 
                default_values["$"]
            )
        
        # Extract only the required metrics, ignoring extra fields
        return (
            data.get("CU", 1.0),
            data.get("EU", 0.0001),
            data.get("CO2", 0.00005),
            data.get("$", 0.00001)
        )


class EnhancedCostAnalyzer:
    """
    Enhanced analyzer class for evaluating algorithm costs with composite scoring.
    
    Provides methods to analyze Python functions, LLVM IR, and PTX code
    to estimate computational costs across multiple metrics, including
    unified composite scores for algorithm comparison.
    """
    
    def __init__(self, arch: str = "x86", 
                 composite_weights: Optional[Dict[str, float]] = None,
                 profile: Optional[str] = None) -> None:
        """
        Initialize enhanced cost analyzer with specified architecture.
        
        Args:
            arch: Target architecture for cost model
            composite_weights: Custom weights for composite score calculation
            profile: Predefined profile name ('HPC', 'MOBILE', 'COMMERCIAL', 'RESEARCH', 'DEFAULT')
        """
        self.model = InstructionCostModel(arch=arch)
        self.composite_calculator = CompositeScoreCalculator(
            weights=composite_weights, 
            profile=profile
        )
        self.benchmark_history = []

    def analyze_function(self, fn: Callable[..., Any], 
                        include_composite: bool = True) -> Dict[str, float]:
        """
        Analyze Python function bytecode and calculate costs with composite score.
        
        Args:
            fn: Python function to analyze
            include_composite: Whether to calculate composite score
            
        Returns:
            Dictionary with cost metrics and composite score
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
        
        Args:
            ir_path: Path to LLVM IR file (.ll)
            include_composite: Whether to calculate composite score
            
        Returns:
            Dictionary with cost metrics and composite score
            
        Raises:
            FileNotFoundError: If IR file cannot be found
        """
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}
        
        with open(ir_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                    
                tokens = line.split()
                if tokens:
                    # Find actual instruction (skip assignments like %1 =)
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
        
        Args:
            ptx_path: Path to PTX file (.ptx)
            include_composite: Whether to calculate composite score
            
        Returns:
            Dictionary with cost metrics and composite score
            
        Raises:
            FileNotFoundError: If PTX file cannot be found
        """
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}
        
        with open(ptx_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if (not line or line.startswith(';') or line.startswith('//') or
                    line.startswith('.') or line.startswith('{')):
                    continue
                    
                tokens = line.split()
                if tokens:
                    instr = tokens[0].upper().rstrip(':')
                    # Skip labels and directives
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
        
        Returns:
            Carbon intensity in kgCO2/kWh, or fallback value if API unavailable
        """
        try:
            response = requests.get(CARBON_INTENSITY_API, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Convert gCO2 to kgCO2
                return data["data"][0]["intensity"]["actual"] / 1000.0
        except (requests.RequestException, KeyError, IndexError):
            pass
            
        return DEFAULT_CARBON_INTENSITY

    def compare_functions(
        self, 
        fn_old: Callable[..., Any], 
        fn_new: Callable[..., Any],
        include_composite: bool = True
    ) -> Dict[str, Any]:
        """
        Compare costs between two functions (differential analysis) with composite scores.
        
        Args:
            fn_old: Original function
            fn_new: New function to compare against
            include_composite: Whether to include composite score analysis
            
        Returns:
            Dictionary with detailed comparison results
        """
        old_cost = self.analyze_function(fn_old, include_composite)
        new_cost = self.analyze_function(fn_new, include_composite)
        
        # Calculate raw differences
        differences = {}
        for key in ["CU", "EU", "CO2", "$"]:
            differences[f"{key}_diff"] = new_cost[key] - old_cost[key]
            differences[f"{key}_ratio"] = (new_cost[key] / old_cost[key]) if old_cost[key] > 0 else float('inf')
            differences[f"{key}_percent_change"] = (
                ((new_cost[key] - old_cost[key]) / old_cost[key]) * 100
                if old_cost[key] > 0 else float('inf')
            )
        
        # Composite score comparison
        if include_composite:
            differences["COMPOSITE_SCORE_diff"] = (
                new_cost["COMPOSITE_SCORE"] - old_cost["COMPOSITE_SCORE"]
            )
            differences["improvement"] = new_cost["COMPOSITE_SCORE"] > old_cost["COMPOSITE_SCORE"]
            differences["old_grade"] = old_cost["SCORE_GRADE"]
            differences["new_grade"] = new_cost["SCORE_GRADE"]
        
        return {
            "old_metrics": old_cost,
            "new_metrics": new_cost,
            "comparison": differences
        }

    def benchmark_suite(self, functions: List[Tuple[str, Callable[..., Any]]]) -> Dict[str, Any]:
        """
        Run benchmark suite on multiple functions and update reference values.
        
        Args:
            functions: List of (name, function) tuples to benchmark
            
        Returns:
            Dictionary with benchmark results and statistics
        """
        results = {}
        raw_metrics = []
        
        for name, func in functions:
            result = self.analyze_function(func, include_composite=True)
            results[name] = result
            raw_metrics.append({k: v for k, v in result.items() if k in ["CU", "EU", "CO2", "$"]})
        
        # Update reference values based on benchmark results
        self.composite_calculator.update_reference_values(raw_metrics)
        self.benchmark_history.extend(raw_metrics)
        
        # Calculate benchmark statistics
        stats = self._calculate_benchmark_stats(results)
        
        return {
            "results": results,
            "statistics": stats,
            "updated_references": self.composite_calculator.reference_values
        }

    def _calculate_benchmark_stats(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate statistical summary of benchmark results."""
        composite_scores = [result["COMPOSITE_SCORE"] for result in results.values()]
        
        return {
            "best_algorithm": max(results.keys(), key=lambda k: results[k]["COMPOSITE_SCORE"]),
            "worst_algorithm": min(results.keys(), key=lambda k: results[k]["COMPOSITE_SCORE"]),
            "average_composite_score": statistics.mean(composite_scores),
            "median_composite_score": statistics.median(composite_scores),
            "composite_score_std": statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0,
            "score_range": max(composite_scores) - min(composite_scores)
        }


def save_enhanced_csv(data: Dict[str, Any], filename: str, report_dir: str) -> None:
    """
    Save enhanced cost data to CSV file with composite scores.
    
    Args:
        data: Dictionary with cost metrics and composite scores
        filename: Output filename
        report_dir: Directory to save reports
    """
    filepath = os.path.join(report_dir, filename)
    with open(filepath, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value", "Description"])
        
        # Basic metrics
        basic_metrics = ["CU", "EU", "CO2", "$"]
        for key in basic_metrics:
            if key in data:
                writer.writerow([key, data[key], f"Raw {key} value"])
        
        # Normalized scores
        for key in basic_metrics:
            normalized_key = f"{key}_normalized"
            if normalized_key in data:
                writer.writerow([normalized_key, data[normalized_key], f"Normalized {key} score (0-100)"])
        
        # Composite metrics
        if "COMPOSITE_SCORE" in data:
            writer.writerow(["COMPOSITE_SCORE", data["COMPOSITE_SCORE"], "Unified composite score (0-100)"])
        if "SCORE_GRADE" in data:
            writer.writerow(["SCORE_GRADE", data["SCORE_GRADE"], "Letter grade rating"])
        if "EFFICIENCY_RATING" in data:
            writer.writerow(["EFFICIENCY_RATING", data["EFFICIENCY_RATING"], "Efficiency rating"])


def create_enhanced_comparison_chart(
    result1: Dict[str, float], 
    result2: Dict[str, float], 
    report_dir: str,
    names: Tuple[str, str] = ("Algorithm v1", "Algorithm v2")
) -> None:
    """
    Create enhanced comparison chart including composite scores.
    
    Args:
        result1: Cost metrics for first algorithm
        result2: Cost metrics for second algorithm
        report_dir: Directory to save chart
        names: Names for the two algorithms
    """
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw metrics comparison
    raw_metrics = ["CU", "EU", "CO2", "$"]
    values1_raw = [result1.get(m, 0) for m in raw_metrics]
    values2_raw = [result2.get(m, 0) for m in raw_metrics]
    
    x = range(len(raw_metrics))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], values1_raw, width=width, label=names[0], alpha=0.8)
    ax1.bar([i + width/2 for i in x], values2_raw, width=width, label=names[1], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(raw_metrics)
    ax1.set_ylabel("Raw Values")
    ax1.set_title("Raw Metrics Comparison")
    ax1.legend()
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Normalized scores comparison
    normalized_metrics = [f"{m}_normalized" for m in raw_metrics]
    values1_norm = [result1.get(m, 50) for m in normalized_metrics]
    values2_norm = [result2.get(m, 50) for m in normalized_metrics]
    
    ax2.bar([i - width/2 for i in x], values1_norm, width=width, label=names[0], alpha=0.8)
    ax2.bar([i + width/2 for i in x], values2_norm, width=width, label=names[1], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_normalized', '') for m in normalized_metrics])
    ax2.set_ylabel("Normalized Scores (0-100)")
    ax2.set_title("Normalized Scores Comparison")
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # Composite score comparison
    composite1 = result1.get("COMPOSITE_SCORE", 50)
    composite2 = result2.get("COMPOSITE_SCORE", 50)
    
    ax3.bar([names[0], names[1]], [composite1, composite2], 
            color=['skyblue', 'lightcoral'], alpha=0.8)
    ax3.set_ylabel("Composite Score (0-100)")
    ax3.set_title("Composite Score Comparison")
    ax3.set_ylim(0, 100)
    
    # Add score values on bars
    ax3.text(0, composite1 + 2, f"{composite1:.1f}", ha='center', va='bottom')
    ax3.text(1, composite2 + 2, f"{composite2:.1f}", ha='center', va='bottom')
    
    # Radar chart for multi-dimensional comparison
    angles = np.linspace(0, 2 * np.pi, len(raw_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    values1_radar = values1_norm + [values1_norm[0]]
    values2_radar = values2_norm + [values2_norm[0]]
    
    ax4.plot(angles, values1_radar, 'o-', linewidth=2, label=names[0])
    ax4.fill(angles, values1_radar, alpha=0.25)
    ax4.plot(angles, values2_radar, 'o-', linewidth=2, label=names[1])
    ax4.fill(angles, values2_radar, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(raw_metrics)
    ax4.set_ylim(0, 100)
    ax4.set_title("Multi-dimensional Performance Radar")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "enhanced_comparison_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_benchmark_summary_chart(benchmark_results: Dict[str, Any], report_dir: str) -> None:
    """
    Create summary chart for benchmark suite results.
    
    Args:
        benchmark_results: Results from benchmark_suite method
        report_dir: Directory to save chart
    """
    results = benchmark_results["results"]
    stats = benchmark_results["statistics"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Composite scores ranking
    algorithms = list(results.keys())
    composite_scores = [results[alg]["COMPOSITE_SCORE"] for alg in algorithms]
    grades = [results[alg]["SCORE_GRADE"] for alg in algorithms]
    
    # Sort by composite score
    sorted_data = sorted(zip(algorithms, composite_scores, grades), key=lambda x: x[1], reverse=True)
    algorithms, composite_scores, grades = zip(*sorted_data)
    
    colors = ['gold' if score >= 85 else 'silver' if score >= 70 else 'bronze' if score >= 55 else 'lightcoral' 
              for score in composite_scores]
    
    bars = ax1.barh(algorithms, composite_scores, color=colors, alpha=0.8)
    ax1.set_xlabel("Composite Score (0-100)")
    ax1.set_title("Algorithm Performance Ranking")
    ax1.set_xlim(0, 100)
    
    # Add score values and grades on bars
    for i, (bar, score, grade) in enumerate(zip(bars, composite_scores, grades)):
        ax1.text(score + 1, i, f"{score:.1f} ({grade})", va='center', ha='left')
    
    # Performance distribution
    all_metrics = ["CU", "EU", "CO2", "$"]
    metric_averages = []
    
    for metric in all_metrics:
        values = [results[alg][f"{metric}_normalized"] for alg in results.keys()]
        metric_averages.append(np.mean(values))
    
    ax2.pie(metric_averages, labels=all_metrics, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Average Performance Distribution by Metric")
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "benchmark_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    """Enhanced main execution function with composite scoring examples."""
    # Architecture detection and analyzer initialization
    detected_arch = platform.machine().lower()
    print(f"[Init] Detected architecture: {detected_arch}")

    # Example algorithms for comparison with different complexity profiles
    def algorithm_linear(n: int) -> int:
        """Linear time algorithm - O(n)."""
        return sum(i for i in range(n))

    def algorithm_constant(n: int) -> int:
        """Constant time algorithm - O(1)."""
        return n * (n - 1) // 2
    
    def algorithm_quadratic(n: int) -> int:
        """Quadratic time algorithm - O(n²)."""
        total = 0
        for i in range(n):
            for j in range(i):
                total += i * j
        return total
    
    def algorithm_recursive(n: int) -> int:
        """Recursive algorithm with higher call overhead."""
        if n <= 1:
            return n
        return algorithm_recursive(n-1) + algorithm_recursive(n-2)

    # Initialize enhanced analyzer with profile-based weights
    print(f"[Config] Available profiles: {list(PROFILE_WEIGHTS.keys())}")

    # Example: Use RESEARCH profile for academic/research scenarios
    analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile="RESEARCH")

    # Print current profile info
    profile_info = analyzer.composite_calculator.get_profile_info()
    print(f"[Config] Using profile: {profile_info['profile']}")
    print(f"[Config] Profile description: {profile_info['description']}")
    print(f"[Config] Weights: {profile_info['weights']}")

    # Single function analysis with composite scoring
    print("\n" + "="*60)
    print("ENHANCED ALGORITHM ANALYSIS WITH COMPOSITE SCORING")
    print("="*60)
    
    result_linear = analyzer.analyze_function(algorithm_linear)
    result_constant = analyzer.analyze_function(algorithm_constant)
    
    print(f"\n[Analysis] Linear Algorithm Assessment:")
    for key, value in result_linear.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n[Analysis] Constant Algorithm Assessment:")
    for key, value in result_constant.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Detailed function comparison
    print(f"\n[Comparison] Detailed Algorithm Comparison:")
    comparison = analyzer.compare_functions(algorithm_linear, algorithm_constant)
    
    print(f"  Original Algorithm:")
    print(f"    Composite Score: {comparison['old_metrics']['COMPOSITE_SCORE']:.2f} ({comparison['old_metrics']['SCORE_GRADE']})")
    print(f"    Efficiency Rating: {comparison['old_metrics']['EFFICIENCY_RATING']}")
    
    print(f"  Optimized Algorithm:")
    print(f"    Composite Score: {comparison['new_metrics']['COMPOSITE_SCORE']:.2f} ({comparison['new_metrics']['SCORE_GRADE']})")
    print(f"    Efficiency Rating: {comparison['new_metrics']['EFFICIENCY_RATING']}")
    
    print(f"  Improvement Analysis:")
    print(f"    Score Improvement: {comparison['comparison']['COMPOSITE_SCORE_diff']:.2f} points")
    print(f"    Better Algorithm: {'Yes' if comparison['comparison']['improvement'] else 'No'}")
    
    for metric in ["CU", "EU", "CO2", "$"]:
        change = comparison['comparison'][f'{metric}_percent_change']
        print(f"    {metric} Change: {change:.1f}%")

    # Benchmark suite analysis
    print(f"\n[Benchmark] Running Algorithm Suite Analysis:")
    
    algorithms_suite = [
        ("Linear_O(n)", algorithm_linear),
        ("Constant_O(1)", algorithm_constant),
        ("Quadratic_O(n²)", algorithm_quadratic),
        ("Recursive_Fib", algorithm_recursive)
    ]
    
    benchmark_results = analyzer.benchmark_suite(algorithms_suite)
    
    print(f"  Best Algorithm: {benchmark_results['statistics']['best_algorithm']}")
    print(f"  Worst Algorithm: {benchmark_results['statistics']['worst_algorithm']}")
    print(f"  Average Composite Score: {benchmark_results['statistics']['average_composite_score']:.2f}")
    print(f"  Score Standard Deviation: {benchmark_results['statistics']['composite_score_std']:.2f}")
    print(f"  Performance Range: {benchmark_results['statistics']['score_range']:.2f} points")

    # Updated reference values after benchmarking
    print(f"\n[Calibration] Updated Reference Values:")
    for metric, refs in benchmark_results['updated_references'].items():
        print(f"  {metric}: min={refs['min']:.6f}, max={refs['max']:.6f}, typical={refs['typical']:.6f}")

    # Profile comparison example
    print(f"\n[Profile Comparison] Testing different profiles:")
    profiles_to_test = ["HPC", "MOBILE", "COMMERCIAL"]
    for test_profile in profiles_to_test:
        test_analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile=test_profile)
        test_result = test_analyzer.analyze_function(algorithm_constant)
        print(f"  {test_profile}: Composite Score = {test_result['COMPOSITE_SCORE']:.2f} ({test_result['SCORE_GRADE']})")

    # Analyze external files if available
    llvm_path = "examples/sample.ll"
    if os.path.exists(llvm_path):
        llvm_result = analyzer.analyze_llvm_ir(llvm_path)
        print(f"\n[LLVM] Assessment of '{llvm_path}':")
        print(f"  Composite Score: {llvm_result['COMPOSITE_SCORE']:.2f} ({llvm_result['SCORE_GRADE']})")

    ptx_path = "examples/sample.ptx"
    if os.path.exists(ptx_path):
        ptx_result = analyzer.analyze_ptx(ptx_path)
        print(f"\n[PTX] Assessment of '{ptx_path}':")
        print(f"  Composite Score: {ptx_result['COMPOSITE_SCORE']:.2f} ({ptx_result['SCORE_GRADE']})")

    # Environmental impact analysis
    carbon_intensity = analyzer.fetch_carbon_intensity()
    print(f"\n[Environment] Current carbon intensity: {carbon_intensity:.6f} kgCO2/kWh")

    # Save enhanced reports
    report_dir = "enhanced_reports"
    os.makedirs(report_dir, exist_ok=True)

    # Save individual results
    save_enhanced_csv(result_linear, "linear_algorithm_enhanced.csv", report_dir)
    save_enhanced_csv(result_constant, "constant_algorithm_enhanced.csv", report_dir)
    
    # Save comparison results
    with open(os.path.join(report_dir, "detailed_comparison.json"), "w", encoding='utf-8') as f:
        json.dump(comparison, f, indent=4)
    
    # Save benchmark results
    with open(os.path.join(report_dir, "benchmark_suite_results.json"), "w", encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=4)

    # Create enhanced visualizations
    create_enhanced_comparison_chart(
        result_linear, result_constant, report_dir,
        names=("Linear O(n)", "Constant O(1)")
    )
    
    create_benchmark_summary_chart(benchmark_results, report_dir)

    # Generate comprehensive summary report
    summary_report = {
        "analysis_timestamp": platform.platform(),
        "architecture": detected_arch,
        "composite_weights": custom_weights,
        "best_single_algorithm": {
            "name": "Constant O(1)",
            "composite_score": result_constant["COMPOSITE_SCORE"],
            "grade": result_constant["SCORE_GRADE"]
        },
        "benchmark_summary": benchmark_results["statistics"],
        "recommendations": generate_recommendations(benchmark_results, profile_info)
    }
    
    with open(os.path.join(report_dir, "comprehensive_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary_report, f, indent=4)

    print(f"\n[Complete] Enhanced reports saved to '{report_dir}' directory")
    print(f"[Complete] Generated files:")
    print(f"  - Individual algorithm assessments (CSV)")
    print(f"  - Detailed comparison analysis (JSON)")
    print(f"  - Benchmark suite results (JSON)")
    print(f"  - Enhanced comparison charts (PNG)")
    print(f"  - Benchmark summary visualization (PNG)")
    print(f"  - Comprehensive summary report (JSON)")


def generate_recommendations(
    benchmark_results: Dict[str, Any], 
    profile_info: Optional[Dict[str, Any]] = None    
) -> List[str]:
    """
    Generate algorithm optimization recommendations based on benchmark results.
    
    Args:
        benchmark_results: Results from benchmark suite
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    stats = benchmark_results["statistics"]
    results = benchmark_results["results"]

    # Add profile-specific recommendations
    if profile_info:
        recommendations.append(
            f"Analysis performed using '{profile_info['profile']}' profile: "
            f"{profile_info['description']}"
        )
    
    best_alg = stats["best_algorithm"]
    worst_alg = stats["worst_algorithm"]
    
    recommendations.append(
        f"Use '{best_alg}' for optimal performance with composite score of "
        f"{results[best_alg]['COMPOSITE_SCORE']:.1f}"
    )
    
    recommendations.append(
        f"Avoid '{worst_alg}' due to poor performance with composite score of "
        f"{results[worst_alg]['COMPOSITE_SCORE']:.1f}"
    )
    
    # Find algorithms with good energy efficiency
    energy_efficient = min(results.keys(), 
                          key=lambda k: results[k].get('EU_normalized', 50))
    recommendations.append(
        f"For energy-critical applications, consider '{energy_efficient}' "
        f"with highest energy efficiency score"
    )
    
    # Find algorithms with low environmental impact
    eco_friendly = min(results.keys(), 
                      key=lambda k: results[k].get('CO2_normalized', 50))
    recommendations.append(
        f"For environmentally conscious deployment, '{eco_friendly}' "
        f"has the lowest carbon footprint"
    )
    
    if stats["score_range"] > 20:
        recommendations.append(
            "Large performance variation detected - consider algorithm selection "
            "based on specific use case requirements"
        )
    
    return recommendations


if __name__ == "__main__":
    main()
    

