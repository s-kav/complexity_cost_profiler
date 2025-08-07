# composite_score_calculator.py

import copy
import json
import math
import statistics
from typing import Dict, Optional, Any, List
from types import MappingProxyType

# --- Configuration Loading ---

DEFAULT_COST_MODEL_PATH = "./cost_models"
DEFAULT_CONFIG_WEIGHTS_PATH = DEFAULT_COST_MODEL_PATH + '/config_weights.json'

def load_configuration(config_file: str = DEFAULT_CONFIG_WEIGHTS_PATH) -> tuple:
    """
    Loads configuration from JSON file and creates constant dictionaries.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        tuple: Tuple with all constant dictionaries
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        weight_profiles = config_data['weight_profiles']
        
        def extract_weights(profile_data):
            if isinstance(profile_data, dict) and 'weights' in profile_data:
                return profile_data['weights']
            return profile_data
        
        DEFAULT_COMPOSITE_WEIGHTS = extract_weights(weight_profiles['default_composite'])
        RESEARCH_WEIGHTS = extract_weights(weight_profiles['research'])
        COMMERCIAL_WEIGHTS = extract_weights(weight_profiles['commercial'])
        MOBILE_WEIGHTS = extract_weights(weight_profiles['mobile'])
        HPC_WEIGHTS = extract_weights(weight_profiles['hpc'])
        
        reference_data = config_data['reference_values']
        REFERENCE_VALUES = reference_data.get('values', reference_data)
        
        return (
            DEFAULT_COMPOSITE_WEIGHTS,
            RESEARCH_WEIGHTS,
            COMMERCIAL_WEIGHTS,
            MOBILE_WEIGHTS,
            HPC_WEIGHTS,
            REFERENCE_VALUES
        )
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found. Please ensure it exists.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in configuration file '{config_file}': {e}")
    except KeyError as e:
        raise KeyError(f"Missing required configuration key in '{config_file}': {e}")

# --- Load configuration at module startup ---
(DEFAULT_COMPOSITE_WEIGHTS, RESEARCH_WEIGHTS, COMMERCIAL_WEIGHTS, 
  MOBILE_WEIGHTS, HPC_WEIGHTS, REFERENCE_VALUES) = load_configuration()

# Create the final mapping that the CompositeScoreCalculator will use
PROFILE_WEIGHTS = MappingProxyType({
    "RESEARCH": RESEARCH_WEIGHTS,
    "COMMERCIAL": COMMERCIAL_WEIGHTS,
    "MOBILE": MOBILE_WEIGHTS,
    "HPC": HPC_WEIGHTS,
    "DEFAULT": DEFAULT_COMPOSITE_WEIGHTS,
})


class CompositeScoreCalculator:
    """
    Calculator for unified composite scores based on multiple cost metrics.
    It now uses configuration loaded from an external JSON file.
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 profile: str = "DEFAULT",
                 reference_values: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize composite score calculator.

        Args:
            weights: Weight distribution for metrics (must sum to 1.0)
            profile: Predefined profile name ('HPC', 'MOBILE', 'COMMERCIAL', 'RESEARCH', 'DEFAULT')
            reference_values: Reference values for normalization. If None, uses values from config file.
        """
        self.profile = profile

        if weights is not None:
            self.weights = dict(weights)
        elif profile is not None:
            if profile not in PROFILE_WEIGHTS:
                raise ValueError(f"Unknown profile: {profile}. Available: {list(PROFILE_WEIGHTS.keys())}")
            # Use the globally loaded weights from the JSON file
            self.weights = PROFILE_WEIGHTS[profile].copy()
        else:
            self.weights = DEFAULT_COMPOSITE_WEIGHTS.copy()

        # Use globally loaded reference values by default
        self.reference_values = copy.deepcopy(reference_values) if reference_values else copy.deepcopy(REFERENCE_VALUES)

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    # ... (the rest of the CompositeScoreCalculator class methods remain EXACTLY THE SAME) ...
    # normalize_metric, _z_to_percentile, calculate_composite_score, _get_score_grade, etc.
    def normalize_metric(self, value: float, metric: str, method: str = "minmax") -> float:
        if metric not in self.reference_values:
            return 50.0
        ref = self.reference_values[metric]
        min_val, max_val = ref["min"], ref["max"]
        if method == "minmax":
            if max_val <= min_val: return 50.0
            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(100.0, 100.0 * (1.0 - normalized)))
        elif method == "zscore":
            typical = ref["typical"]
            std_estimate = (max_val - min_val) / 6
            if std_estimate <= 0: return 50.0
            z_score = (value - typical) / std_estimate
            percentile = self._z_to_percentile(-z_score)
            return max(0.0, min(100.0, percentile))
        elif method == "log":
            if value <= 0 or min_val <= 0 or max_val <= min_val: return 50.0
            log_val = math.log(value)
            log_min, log_max = math.log(min_val), math.log(max_val)
            normalized = (log_val - log_min) / (log_max - log_min)
            return max(0.0, min(100.0, 100.0 * (1.0 - normalized)))
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _z_to_percentile(self, z_score: float) -> float:
        return 50.0 * (1.0 + math.erf(z_score / math.sqrt(2)))

    def calculate_composite_score(self, metrics: Dict[str, float], method: str = "minmax") -> Dict[str, float]:
        normalized_scores = {}
        for metric, value in metrics.items():
            if metric in self.weights:
                normalized_scores[f"{metric}_normalized"] = self.normalize_metric(value, metric, method)
        composite_score = sum(weight * normalized_scores.get(f"{metric}_normalized", 0) for metric, weight in self.weights.items())
        result = metrics.copy()
        result.update(normalized_scores)
        result["COMPOSITE_SCORE"] = composite_score
        result["SCORE_GRADE"] = self._get_score_grade(composite_score)
        result["EFFICIENCY_RATING"] = self._get_efficiency_rating(composite_score)
        return result

    def _get_score_grade(self, score: float) -> str:
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
        if score >= 85: return "Excellent"
        elif score >= 70: return "Good"
        elif score >= 55: return "Average"
        elif score >= 40: return "Below Average"
        else: return "Poor"

    def update_reference_values(self, benchmark_results: List[Dict[str, float]]) -> None:
        if not benchmark_results: return
        for metric in self.weights.keys():
            values = [result.get(metric, 0) for result in benchmark_results if metric in result]
            if values:
                self.reference_values[metric] = {"min": min(values), "max": max(values), "typical": statistics.median(values)}

    def get_profile_info(self) -> Dict[str, Any]:
        return {"profile": self.profile, "weights": self.weights.copy(), "description": self._get_profile_description()}

    def _get_profile_description(self) -> str:
        # We can dynamically get descriptions from a loaded config in the future,
        # but for now, this remains a simple and effective way.
        descriptions = {
            "HPC": "High Performance Computing - optimized for maximum computational throughput",
            "MOBILE": "Mobile/IoT - optimized for energy efficiency and battery life",
            "COMMERCIAL": "Commercial Cloud - balanced approach with cost consideration",
            "RESEARCH": "Research/Academic - focused on performance with environmental awareness",
            "DEFAULT": "Default balanced profile for general use cases",
            "CUSTOM": "Custom weight configuration"
        }
        return descriptions.get(self.profile, "Custom profile configuration")


class WeightConfiguration:
    """Class for managing weight coefficient configuration."""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG_WEIGHTS_PATH):
        self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> None:
        """Loads configuration from file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create class attributes
        profiles = config_data['weight_profiles']
        self.DEFAULT_COMPOSITE_WEIGHTS = profiles['default_composite']
        self.RESEARCH_WEIGHTS = profiles['research']
        self.COMMERCIAL_WEIGHTS = profiles['commercial']
        self.MOBILE_WEIGHTS = profiles['mobile']
        self.HPC_WEIGHTS = profiles['hpc']
        self.REFERENCE_VALUES = config_data['reference_values']
    
    def get_profile(self, profile_name: str) -> Dict[str, float]:
        """Returns weight profile by name."""
        profile_mapping = {
            'default_composite': self.DEFAULT_COMPOSITE_WEIGHTS,
            'research': self.RESEARCH_WEIGHTS,
            'commercial': self.COMMERCIAL_WEIGHTS,
            'mobile': self.MOBILE_WEIGHTS,
            'hpc': self.HPC_WEIGHTS
        }
        
        if profile_name not in profile_mapping:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        return profile_mapping[profile_name]
    
    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validates that weight sum equals 1.0 and all keys are present."""
        required_keys = {'CU', 'EU', 'CO2', '$'}
        
        if set(weights.keys()) != required_keys:
            return False
        
        total_weight = sum(weights.values())
        return abs(total_weight - 1.0) < 1e-10