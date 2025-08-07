
# Complexity Cost Profiler

CostComplexityProfiler: Advanced Algorithmic Complexity Assessment Model with Unified Composite Score.
A sophisticated static analysis tool designed to evaluate the "cost" of software projects across multiple dimensions beyond traditional time complexity.

## The Core Problem

Traditional algorithm analysis, often limited to Big O notation, is abstract and fails to capture critical real-world operational costs. In modern software development, factors like energy consumption (for mobile/IoT), cloud infrastructure expenses, and environmental impact are crucial considerations.

The **Complexity Cost Profiler** addresses this gap by providing a multi-faceted cost assessment, allowing developers and managers to make more informed decisions based on a holistic view of software efficiency.

## Purpose 

This enhanced prototype implements a Python utility for evaluating computational "cost" of algorithms considering processor architectural features (CPU/ARM/GPU), energy consumption metrics and the ability to translate costs into monetary equivalent. Now includes a unified composite score based on scaled normalization of 4 main metrics and 5 profiles.

## Key Features

-   **Multi-Dimensional Analysis**: Evaluates code based on four key metrics:
    -   **CU (Computational Units)**: Abstract measure of computational effort.
    -   **EU (Energy Units)**: Estimated energy consumption.
    -   **CO2 (CO2 Units)**: Estimated carbon footprint.
    -   **$ (Monetary Units)**: Estimated monetary cost for execution (e.g., in a cloud environment).
-   **Composite Score**: Combines the four core metrics into a single, unified score (0-100) for easy, high-level comparison. A higher score indicates better overall efficiency.
-   **Analysis Profiles**: Applies different weighting schemes to the metrics, allowing for cost assessment tailored to specific deployment targets:
    -   `HPC`: Prioritizes computational performance (CU).
    -   `MOBILE`: Prioritizes energy efficiency (EU).
    -   `COMMERCIAL`: Balances performance with monetary cost ($).
    -   `RESEARCH`: Focuses on performance while being mindful of environmental impact.
    -   `DEFAULT`: A balanced profile for general use.
-   **Automated Repository Analysis**: Can clone one or more Git repositories, automatically discover source files (`.py`, `.ll`, `.ptx`), and perform a comprehensive analysis.
-   **Extensible Cost Models**: Instruction costs for different architectures (e.g., `x86_64`) are defined in simple JSON files, making the system easy to extend and calibrate.
-   **Rich Visualizations**: Generates insightful charts to visually compare the efficiency of different algorithms or repositories.

## How It Works

The profiler follows a systematic pipeline for each analysis:

1.  **Code Acquisition**: Clones a Git repository or targets local source files.
2.  **Disassembly**: Statically analyzes source files, breaking down functions and methods into fundamental instructions (e.g., Python bytecode).
3.  **Cost Aggregation**: For each instruction, it looks up the corresponding costs (CU, EU, CO2, $) from an architecture-specific JSON model. These costs are summed up for the entire codebase.
4.  **Normalization & Scoring**: The aggregated raw costs are normalized to a 0-100 scale. Using the weights from a selected **Profile**, it calculates the final **Composite Score**.
5.  **Reporting**: Generates detailed summary tables and comparison charts to present the findings.

## Formulas: Computational Cost Analysis Framework

### Variable Definitions

Let:
- `op_i`: number of instructions of type `i`
- `w_i`: weight (cost) of instruction `i` in CU
- `f_e(i)`: energy consumption of instruction `i` (in Joules)
- `f_c(i)`: CO₂ footprint (kg CO₂)
- `f_d(i)`: monetary cost of executing instruction `i` ($)

### Total Cost Calculations

#### Computational Units
```
COST_total = ∑(op_i × w_i)       [in CU]
```

#### Energy Consumption
```
ENERGY_total = ∑(op_i × f_e(i))  [in Joules]
```

#### Carbon Footprint
```
CO2_total = ∑(op_i × f_c(i))     [in kg CO₂]
```

#### Monetary Cost
```
MONEY_total = ∑(op_i × f_d(i))   [in $ or €]
```

### Composite Score Formula

Let `S_cu`, `S_eu`, `S_co2`, `S_$` be normalized scores (0-100) for each metric.


Then:
```
COMPOSITE_SCORE = α×S_cu + β×S_eu + γ×S_co2 + δ×S_$
```

**Constraint:**
```
α + β + γ + δ = 1 (configurable weights)
```

---

*Note: All weights (α, β, γ, δ) are user-configurable to prioritize different optimization objectives.*

## Usage Examples

The tool is designed to be run from a script. The main workflow involves defining a list of target repositories and launching the analysis loop.

```python
# --- Main execution loop for multiple repositories ---
from IPython.display import display, Image # For displaying charts in notebooks
import platform
import os
import subprocess
import shutil

import utils as u


###########################################
####            main body              ####
###########################################

print("\n"*2 + '#'*150 + "\n")
print("\t"*6 + "\033[1mCOMPREHENSIVE REPOSITORY ANALYSIS\033[0m")

# 1. List of URL-addresses of  repos
REPO_URLS = [
    "https://github.com/s-kav/ds_tools.git",
    "https://github.com/s-kav/kz_data_imputation.git",
    "https://github.com/s-kav/s3_s4_activation_function.git",
    "https://github.com/egorsmkv/radtts-uk-vocos-demo.git",
    "https://github.com/PINTO0309/yolov9_wholebody34_heatmap_vis.git",
    "https://github.com/MaAI-Kyoto/MaAI.git",
    "https://github.com/TheAlgorithms/Python.git",
    "https://github.com/tweepy/tweepy.git",
    "https://github.com/lincolnloop/python-qrcode.git",
    "https://github.com/prompt-toolkit/python-prompt-toolkit.git"
]
# Important! All these repos are opened! Not for advertising!

detected_arch = platform.machine().lower()
all_repo_data = {}

# Analyze each repository and store its data
for repo_url in REPO_URLS:
    local_repo_path = repo_url.split('/')[-1].replace('.git', '')

    if not os.path.isdir(local_repo_path):
        subprocess.run(['git', 'clone', repo_url], capture_output=True, text=True)

    repository_df = u.analyze_repository(
        repo_path=local_repo_path,
        detected_arch=detected_arch,
        verbose=False 
    )
    
    if not repository_df.empty:
        repository_df.set_index('PROFILE NAME', inplace=True)
        all_repo_data[local_repo_path] = repository_df # Store the result
    
        print(f"\n[Analysis] Aggregated assessment for repository: '{local_repo_path}'")
        display(repository_df.style.format({
            'COMPOSITE_SCORE': '{:.2f}',
            'CU': '{:,.0f}',
            'EU': '{:,.4f}',
            'CO2': '{:,.4f}',
            '$': '{:,.4f}',
        }))
```

## Example Output

The analysis of each repository produces a detailed table showing its performance from the perspective of each profile.

**[Analysis] Aggregated assessment for repository: 'kz_data_imputation'**
| PROFILE NAME | File Type | Function Name | COMPOSITE_SCORE | SCORE_GRADE | CU | EU | CO2 | $ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RESEARCH | Python (12) | 89 | 42.23 | D | 12,607 | 1.1194 | 0.3763 | 0.1261 |
| COMMERCIAL | Python (12) | 89 | 30.24 | F | 12,607 | 1.1194 | 0.3763 | 0.1261 |
| MOBILE | Python (12) | 89 | 53.76 | C- | 12,607 | 1.1194 | 0.3763 | 0.1261 |
| HPC | Python (12) | 89 | 36.00 | F | 12,607 | 1.1194 | 0.3763 | 0.1261 |
| DEFAULT | Python (12) | 89 | 37.79 | F | 12,607 | 1.1194 | 0.3763 | 0.1261 |
| TOTAL | All Files (12) | 89 | 40.00 | D | 12,607 | 1.1194 | 0.3763 | 0.1261 |

Finally, a summary chart is generated to compare all analyzed repositories at a glance.

```python
if all_repo_data:
    print("\n[Chart] Generating summary comparison chart for all repositories...")
    
    summary_chart_filepath = os.path.join(u.DEFAULT_REPORT_DIR, "repository_comparison_summary.png")
    
    # Call the new function to create the comparison chart.
    # You can change the profile to plot, e.g., "RESEARCH", "COMMERCIAL", etc.
    u.create_repository_comparison_chart(
        repo_data=all_repo_data,
        output_filepath=summary_chart_filepath,
        profile_to_plot="TOTAL" # This determines which row to use from each DataFrame     
    )
else:
    print("\n[Info] No data was collected from repositories, skipping chart generation.")
```

**[Chart] Displaying generated chart...**

![Repository Comparison Chart](./results/repository_comparison_summary_10_repos.png)


```python
import platform
import os
import glob

import pandas as pd

import utils as u
import benchmark_algorithms as ba
import composite_score_calculator as csc
from enhanced_cost_analyzer import EnhancedCostAnalyzer


###########################################
####            main body              ####
###########################################

DEFAULT_EXAMPLES_PATH = "./examples"

print("\n"*2 + '#'*150 + "\n")
print("\t"*5 + "\033[1mANALYSIS OF EXTERNAL SOURCE FILES (LLVM, PTX, Python)\033[0m")

detected_arch = platform.machine().lower()
analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile="RESEARCH")
profile_info = analyzer.composite_calculator.get_profile_info()

if not os.path.isdir(DEFAULT_EXAMPLES_PATH):
    print(f"\n[Info] Directory '{DEFAULT_EXAMPLES_PATH}' not found. Skipping analysis of external files.")
else:
    # --- Step 1: Collect all raw results from all files and functions ---
    all_individual_results = []
    
    # Define the file types we want to analyze and the function to use for each.
    file_types_to_analyze = {
        'Python': {
            'extension': 'py',
            'analysis_func': analyzer.analyze_py_file # This must be defined in your analyzer class
        },
        'LLVM IR': {
            'extension': 'll',
            'analysis_func': analyzer.analyze_llvm_ir
        },
        'PTX GPU': {
            'extension': 'ptx',
            'analysis_func': analyzer.analyze_ptx
        }
    }

    print(f"\n[Info] Found directory '{DEFAULT_EXAMPLES_PATH}'. Searching for compatible files...")
    
    # Loop through our defined file types and find all matching files.
    for file_type, config in file_types_to_analyze.items():
        search_pattern = os.path.join(DEFAULT_EXAMPLES_PATH, f"**/*.{config['extension']}")
        found_files = glob.glob(search_pattern, recursive=True)

        for file_path in found_files:
            print(f"  > Analyzing {file_type} file: {file_path}")
            
            # Call the appropriate analysis function
            results_from_file = config['analysis_func'](file_path)
            
            # The result can be a list (for .py) or a single dict (for others)
            # To handle both consistently, we'll ensure we have a list.
            if not isinstance(results_from_file, list):
                # For non-python files, wrap the single result dict in a list
                single_result = results_from_file
                single_result['Source File'] = os.path.basename(file_path)
                single_result['Function Name'] = '-' # No specific function name
                single_result['File Type'] = file_type
                results_from_file = [single_result]
            
            # Add the File Type to results from Python files
            if file_type == 'Python':
                for res in results_from_file:
                    res['File Type'] = file_type

            # Use extend to add all items from the list to our main results
            all_individual_results.extend(results_from_file)

    # --- Step 2: Aggregate the results by source file ---
    aggregated_results = {}

    for result in all_individual_results:
        file_name = result['Source File']
        
        if file_name not in aggregated_results:
            # Initialize the entry for this file
            aggregated_results[file_name] = {
                'Source File': file_name,
                'File Type': result.get('File Type', 'Unknown'),
                'Function Name': [], # We will collect function names in a list
                'CU': 0, 'EU': 0, 'CO2': 0, '$': 0
            }
        
        # Aggregate raw numeric metrics
        for metric in ['CU', 'EU', 'CO2', '$']:
            aggregated_results[file_name][metric] += result.get(metric, 0)
        
        # Collect function names, avoiding duplicates and placeholders
        func_name = result.get('Function Name')
        if func_name and func_name not in aggregated_results[file_name]['Function Name']:
            aggregated_results[file_name]['Function Name'].append(func_name)

    # --- Step 3: Finalize the aggregated data for display ---
    final_aggregated_list = []
    for file_name, data in aggregated_results.items():
        # Join the list of function names into a single readable string
        if data['Function Name']:
            data['Function Name'] = ', '.join(sorted(data['Function Name']))
        else:
            # Handle files like .ll or .ptx that have no function names
            data['Function Name'] = '-'
            
        # If the file had an error or no functions, its raw metrics will be 0.
        # We need to handle this to avoid division by zero in score calculation.
        if data['CU'] > 0:
            # Recalculate the composite score based on the SUMMED raw metrics
            final_data_with_score = analyzer.composite_calculator.calculate_composite_score(data)
            final_aggregated_list.append(final_data_with_score)
        else:
            # For files with no analyzable content, just add them with 0 scores
            data['COMPOSITE_SCORE'] = data.get('COMPOSITE_SCORE', 0)
            data['SCORE_GRADE'] = data.get('SCORE_GRADE', 'N/A')
            final_aggregated_list.append(data)

    # --- Step 4: Create and display the final aggregated DataFrame ---
    if not final_aggregated_list:
        print("\n[Info] No compatible files (.py, .ll, .ptx) found in the 'examples' directory.")
    else:
        external_df = pd.DataFrame(final_aggregated_list)
        
        # Define the columns for the final output table
        display_columns = [
            'Source File', 'File Type', 'Function Name',
            'COMPOSITE_SCORE', 'SCORE_GRADE', 'CU', 'EU', 'CO2', '$'
        ]
        # Ensure we only try to access columns that actually exist
        existing_cols = [col for col in display_columns if col in external_df.columns]
        external_df = external_df[existing_cols]

        # Sort the results and set the index for a clean look
        external_df.sort_values(by='COMPOSITE_SCORE', ascending=False, inplace=True)
        external_df.set_index('Source File', inplace=True)
        
        print("\n[Analysis] Aggregated Assessment of External Files:")
        print(external_df) # or display(external_df)
```

**[Chart] Displaying generated chart...**

![Benchmark Suite Summary](./results/benchmark_suite_summary_chart.png)

![Pre-selected algorithms Comparison Chart](./results/Cubic_On3_TripleLoops_vs_Constant_O1_Formula.png)


# References

For citing you should use:

Sergii Kavun. (2025). s-kav/complexity_cost_profiler: version 1.0 (v.1.0). Zenodo. https://doi.org/10.5281/zenodo.16761183

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16761183.svg)](https://doi.org/10.5281/zenodo.16761183)


# License

This project is licensed under the Apache-2.0 license - see the [LICENSE](https://github.com/s-kav/complexity_cost_profiler/blob/main/LICENSE) file for details.


## Future Roadmap

-   **CI/CD Integration**: Develop a GitHub Action to run the profiler on pull requests and report on potential efficiency regressions.
-   **Expanded Language Support**: Add analyzers for other languages like C++, Java, and JavaScript.
-   **Hardware-Based Calibration**: Create tools to generate more accurate cost models by profiling instruction costs on real hardware.
-   **Web Dashboard**: Build a web-based UI to visualize historical analysis data and track project efficiency over time.
