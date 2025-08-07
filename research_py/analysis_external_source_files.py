# analysis_external_source_files.py

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

###########################################
####            plotting               ####
###########################################

if not external_df.empty:
    print("\n[Chart] Generating custom performance scatter plot...")
    
    scatterplot_filepath = os.path.join(u.DEFAULT_REPORT_DIR, "file_performance_scatterplot.png")
    
    # Call the new function. You can change the y-axis metric here if you want.

    u.create_file_performance_scatterplot(
        data=external_df,
        output_filepath=scatterplot_filepath,
        y_axis_metric='CU' # This can be changed to 'EU', 'CO2', or '$'        
    )
