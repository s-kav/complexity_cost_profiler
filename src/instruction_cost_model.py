# instruction_cost_model.py

import os
import json
from typing import Dict, Tuple

# The path constant now lives with the class that uses it.
DEFAULT_COST_MODEL_PATH = "./cost_models"

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
        