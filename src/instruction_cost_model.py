# complexity_cost_profiler/src/instruction_cost_model.py

import os
import json
from typing import Dict, Optional, Tuple

# The path constant now lives with the class that uses it.
DEFAULT_COST_MODEL_PATH = "./cost_models"

# Precision suffix tags embedded in PTX instruction names (e.g. "MUL.F16")
_PRECISION_TAGS = {
    "F64": "FP64",
    "F32": "FP32",
    "F16": "FP16",
    "BF16": "BF16",
    "S8":  "INT8",
    "U8":  "INT8",
}

# Per-precision CU/EU/CO2/$ multipliers relative to FP64 baseline.
# Source: NVIDIA A100 SXM4 throughput data.
PRECISION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "FP64": {"CU": 1.0,    "EU": 1.0,   "CO2": 1.0,   "$": 1.0   },
    "FP32": {"CU": 0.5,    "EU": 0.55,  "CO2": 0.55,  "$": 0.55  },
    "BF16": {"CU": 0.125,  "EU": 0.20,  "CO2": 0.20,  "$": 0.20  },
    "FP16": {"CU": 0.125,  "EU": 0.18,  "CO2": 0.18,  "$": 0.18  },
    "INT8": {"CU": 0.0625, "EU": 0.10,  "CO2": 0.10,  "$": 0.10  },
}

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
            arch: Target architecture (e.g., 'x86_64', 'arm', 'gpu')

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
        model_file = os.path.join(DEFAULT_COST_MODEL_PATH, f"{self.arch}_instr_costs.json")

        try:
            if os.path.exists(model_file):
                with open(model_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"Warning: Could not load cost model from {model_file}. Error: {e}. Using default values.")
            # Fall through to return default model
        
        print(f"Warning: Cost model not found for architecture: {self.arch} at path {model_file}. Using default values.")
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

    def get_cost1(self, opname: str, is_bytecode: bool = False) -> Tuple[float, float, float, float]:
        """
        Get cost metrics for specified instruction.

        Args:
            opname: Instruction operation name
            is_bytecode: Whether the opname is Python bytecode instruction

        Returns:
            Tuple of (CU, EU, CO2, $) cost values
        """
        opname = opname.upper()

        if is_bytecode and opname in self.bytecode_mapping:
            opname = self.bytecode_mapping[opname]

        data = self.weights.get(opname)

        if data is None:
            default_values = {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001}
            return (default_values["CU"], default_values["EU"], default_values["CO2"], default_values["$"])

        return (
            data.get("CU", 1.0),
            data.get("EU", 0.0001),
            data.get("CO2", 0.00005),
            data.get("$", 0.00001)
        )
    def get_cost(self, opname: str, is_bytecode: bool = False,
                 precision: Optional[str] = None) -> Tuple[float, float, float, float]:
        """
        Get cost metrics for specified instruction.

        Args:
            opname:      Instruction operation name.
            is_bytecode: Whether the opname is a Python bytecode instruction.
            precision:   Optional precision override ('FP64', 'FP32', 'BF16',
                         'FP16', 'INT8').  When provided, the base cost is
                         multiplied by the corresponding precision factor.
                         If the instruction name already encodes a precision
                         suffix (e.g. ``MUL.F16``) the suffix takes precedence
                         over this argument.

        Returns:
            Tuple of (CU, EU, CO2, $) cost values.
        """
        opname = opname.upper()

        if is_bytecode and opname in self.bytecode_mapping:
            opname = self.bytecode_mapping[opname]

        data = self.weights.get(opname)

        if data is None:
            default_values = {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001}
            cu, eu, co2, usd = (default_values["CU"], default_values["EU"],
                                default_values["CO2"], default_values["$"])
        else:
            cu  = data.get("CU",  1.0)
            eu  = data.get("EU",  0.0001)
            co2 = data.get("CO2", 0.00005)
            usd = data.get("$",   0.00001)

        # Detect precision from the instruction suffix (e.g. "MUL.F16" → "FP16")
        detected = self._detect_precision_from_opname(opname)
        effective_precision = detected or precision

        if effective_precision and effective_precision in PRECISION_MULTIPLIERS:
            mults = PRECISION_MULTIPLIERS[effective_precision]
            cu  *= mults["CU"]
            eu  *= mults["EU"]
            co2 *= mults["CO2"]
            usd *= mults["$"]

        return (cu, eu, co2, usd)


    @staticmethod
    def _detect_precision_from_opname(opname: str) -> Optional[str]:
        """Return precision string if the opname contains a known precision tag."""
        for tag, precision in _PRECISION_TAGS.items():
            if f".{tag}" in opname or opname.endswith(tag):
                return precision
        return None


    def get_precision_multipliers(self, precision: str) -> Dict[str, float]:
        """Return the raw CU/EU/CO2/$ multipliers for a given precision."""
        if precision not in PRECISION_MULTIPLIERS:
            raise ValueError(f"Unknown precision '{precision}'. Supported: {list(PRECISION_MULTIPLIERS)}")
        return PRECISION_MULTIPLIERS[precision].copy()




