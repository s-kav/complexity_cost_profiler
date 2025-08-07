"""
CostComplexityProfiler: Расширенная модель оценки алгоритмической сложности
-------------------------------------------------------------

Назначение:
Этот прототип реализует утилиту на Python для оценки вычислительной "стоимости"
алгоритмов с учётом архитектурных особенностей процессоров (CPU/ARM/GPU),
метрик энергопотребления и возможности перевода затрат в денежный эквивалент.

Функциональные блоки:
1. Анализ кода на уровне инструкций (Python bytecode, LLVM IR, PTX)
2. Модель стоимости команд (такты, энергия, деньги)
3. Поддержка калибровки под разные архитектуры
4. Оценка эффекта изменения алгоритма (дифференциальный анализ)
5. Метрики: Cost Units (CU), Energy Units (EU), Carbon Units (kgCO2), Dollar Units ($)
6. Расчёт углеродного следа через внешний API (например, carbon-intensity.org)
7. Поддержка расширяемости на ветвления, кэш, параллелизм
8. Поддержка шаблонов моделей ARM/GPU (instr_costs.json)

Формулы:
----------
Let:
- op_i: количество инструкций типа i
- w_i: вес (стоимость) инструкции i в CU
- f_e(i): энергозатрата инструкции i (в Джоулях)
- f_c(i): CO2 footprint (кг CO2)
- f_d(i): денежная стоимость выполнения инструкции i ($)

Тогда:
  COST_total = ∑(op_i × w_i)       [в CU]
  ENERGY_total = ∑(op_i × f_e(i))  [в Джоулях]
  CO2_total = ∑(op_i × f_c(i))     [в кг CO2]
  MONEY_total = ∑(op_i × f_d(i))   [в $ или €]

"""

import dis
import platform
import json
import os
import subprocess
import requests
from typing import Dict, Tuple
import csv
import matplotlib.pyplot as plt
import pprint

CARBON_INTENSITY_API = "https://api.carbonintensity.org.uk/intensity"
DEFAULT_COST_MODEL_PATH = "./cost_models"

class InstructionCostModel:
    def __init__(self, arch: str):
        self.arch = arch.lower()
        self.weights = self._load_weights()

    def _load_weights(self) -> Dict[str, Dict[str, float]]:
        model_file = os.path.join(os.path.dirname(__file__), f"cost_models/{self.arch}_instr_costs.json")
        try:
            with open(model_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Не найдена модель стоимости для архитектуры: {self.arch}")

    def get_cost(self, opname: str) -> Tuple[float, float, float, float]:
        opname = opname.upper()
        data = self.weights.get(opname, None)
        if data is None:
            data = {"CU": 1.0, "EU": 0.0001, "CO2": 0.00005, "$": 0.00001}
        return (data["CU"], data["EU"], data["CO2"], data["$"])

class CostAnalyzer:
    def __init__(self, arch: str = "x86"):
        self.model = InstructionCostModel(arch=arch)

    def analyze_function(self, fn) -> Dict[str, float]:
        instructions = list(dis.get_instructions(fn))
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}
        for instr in instructions:
            cu, eu, co2, money = self.model.get_cost(instr.opname)
            summary["CU"] += cu
            summary["EU"] += eu
            summary["CO2"] += co2
            summary["$"] += money
        return summary

    def analyze_llvm_ir(self, ir_path: str) -> Dict[str, float]:
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}
        with open(ir_path, 'r') as f:
            for line in f:
                opcode = line.strip().split()[0].upper()
                cu, eu, co2, money = self.model.get_cost(opcode)
                summary["CU"] += cu
                summary["EU"] += eu
                summary["CO2"] += co2
                summary["$"] += money
        return summary

    def analyze_ptx(self, ptx_path: str) -> Dict[str, float]:
        summary = {"CU": 0.0, "EU": 0.0, "CO2": 0.0, "$": 0.0}
        with open(ptx_path, 'r') as f:
            for line in f:
                if ";" in line: continue
                tokens = line.strip().split()
                if tokens:
                    instr = tokens[0].upper().strip()
                    cu, eu, co2, money = self.model.get_cost(instr)
                    summary["CU"] += cu
                    summary["EU"] += eu
                    summary["CO2"] += co2
                    summary["$"] += money
        return summary

    def fetch_carbon_intensity(self) -> float:
        try:
            resp = requests.get(CARBON_INTENSITY_API, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["intensity"]["actual"] / 1000.0  # г CO2 → кг CO2
        except Exception:
            pass
        return 0.2  # fallback значение

    def compare_functions(self, fn_old, fn_new) -> Dict[str, float]:
        old_cost = self.analyze_function(fn_old)
        new_cost = self.analyze_function(fn_new)
        return {key: new_cost[key] - old_cost[key] for key in old_cost}

if __name__ == "__main__":
    
    # Выбор архитектуры и инициализация анализатора
    detected_arch = platform.machine().lower()
    print(f"[Init] Определена архитектура: {detected_arch}")

    def algorithm_v1(n): return sum(i for i in range(n))
    def algorithm_v2(n): return n * (n - 1) // 2

    analyzer = CostAnalyzer(arch=detected_arch)

    # Анализ функций Python
    result1 = analyzer.analyze_function(algorithm_v1)
    result2 = analyzer.analyze_function(algorithm_v2)
    diff = analyzer.compare_functions(algorithm_v1, algorithm_v2)

    print("\n[Report] Оценка алгоритма v1:")
    pprint.pprint(result1)
    print("[Report] Оценка алгоритма v2:")
    pprint.pprint(result2)
    print("[Diff] Разница (v2 - v1):")
    pprint.pprint(diff)

    # Анализ LLVM IR (Intermediate Representation between high-level C and machine code)
    llvm_path = "/examples/sample.ll"
    if os.path.exists(llvm_path):
        llvm_result = analyzer.analyze_llvm_ir(llvm_path)
        print("Оценка LLVM IR (sample.ll):", llvm_result)
        print(f"\n[LLVM] Оценка файла '{llvm_path}':")
        pprint.pprint(llvm_result)

    # Анализ PTX (Parallel Thread Execution)
    ptx_path = "/examples/sample.ptx"
    if os.path.exists(ptx_path):
        ptx_result = analyzer.analyze_ptx(ptx_path)
        print("Оценка GPU PTX (sample.ptx):", ptx_result)
        print(f"\n[PTX] Оценка файла '{ptx_path}':")
        pprint.pprint(ptx_result)

    # Получение углеродной интенсивности
    carbon_intensity = analyzer.fetch_carbon_intensity()
    print(f"Текущая углеродная интенсивность (kgCO2/kWh): {carbon_intensity}")

    # Сохранение отчётов в CSV и JSON
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    def save_csv(data, filename):
        with open(os.path.join(report_dir, filename), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for key, value in data.items():
                writer.writerow([key, value])

    def save_json(data, filename):
        with open(os.path.join(report_dir, filename), "w") as f:
            json.dump(data, f, indent=4)

    save_csv(result1, "algorithm_v1.csv")
    save_csv(result2, "algorithm_v2.csv")
    save_csv(diff, "comparison.csv")
    save_json(result1, "algorithm_v1.json")
    save_json(result2, "algorithm_v2.json")
    save_json(diff, "comparison.json")

    # Визуализация сравнения
    labels = list(result1.keys())
    values1 = list(result1.values())
    values2 = list(result2.values())

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x, values1, width=width, label='Algorithm v1')
    plt.bar([p + width for p in x], values2, width=width, label='Algorithm v2')
    plt.xticks([p + width/2 for p in x], labels)
    plt.ylabel("Стоимость")
    plt.title("Сравнение стоимости исполнения алгоритмов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "comparison_chart.png"))
    plt.close()
