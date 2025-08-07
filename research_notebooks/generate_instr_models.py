import os, json

MODELS = {
    "x86": "x86_instr_costs.json",
    "arm": "arm_instr_costs.json",
    "gpu": "gpu_ptx_instr_costs.json"
}

DATA = {
    "x86": [
        {"name": "ADD",    "group": "arith", "CU": 1.0, "EU": 0.0001},
        {"name": "MUL",    "group": "arith", "CU": 2.0, "EU": 0.0002},
        {"name": "LOAD",   "group": "memory","CU": 3.0, "EU": 0.00025, "cache_hit": 0.0001, "cache_miss": 0.0005},
        {"name": "ADDPS",  "group": "simd",  "CU": 2.0, "EU": 0.0003}
    ],
    "arm": [
        {"name": "ADD",    "group": "arith", "CU": 1.0, "EU": 0.00008},
        {"name": "LDR",    "group": "memory","CU": 3.0, "EU": 0.00025, "cache_hit": 0.0001, "cache_miss": 0.0005},
        {"name": "BL",     "group": "control","CU": 2.0, "EU": 0.0002}
    ],
    "gpu": [
        {"name": "ADD.U32", "group": "arith", "CU": 1.0, "EU": 0.0001},
        {"name": "LD.GLOBAL", "group": "memory", "CU": 3.0, "EU": 0.0003, "cache_hit": 0.0001, "cache_miss": 0.0006}
    ]
}

def enrich(instr):
    ci = 0.25  # kgCO2/kWh
    p = 0.12   # $/kWh
    EU = instr["EU"]
    instr["CO2"] = EU * ci / 3600000
    instr["$"] = EU * p / 3600000
    return instr

def generate_models(output_dir="cost_models"):
    os.makedirs(output_dir, exist_ok=True)
    for arch, file in MODELS.items():
        model = {}
        for entry in DATA[arch]:
            model[entry["name"]] = enrich(entry)
            model[entry["name"]]["group"] = entry["group"]
            if "cache_hit" in entry:
                model[entry["name"]]["cache_hit"] = entry["cache_hit"]
                model[entry["name"]]["cache_miss"] = entry["cache_miss"]
        with open(os.path.join(output_dir, file), "w") as f:
            json.dump(model, f, indent=4)
    print("✅ Instructions have been updated.")

if __name__ == "__main__":
    generate_models()
