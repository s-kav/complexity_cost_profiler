# analyzer_configuration_environment.py

import platform
import shutil
import os
import json

import pandas as pd

import utils as u
import benchmark_algorithms as ba
import composite_score_calculator as csc
from enhanced_cost_analyzer import EnhancedCostAnalyzer


###########################################
####            main body              ####
###########################################

SAVING_FLAG = True
DEFAULT_REPORT_DIR = "./enhanced_reports"

print("\n"*2 + '#'*150 + "\n")
print("\t"*2 + "\033[1mANALYZER CONFIGURATION AND ENVIRONMENT\033[0m")

# 1. Perform the necessary initializations (no changes here)
detected_arch = platform.machine().lower()
analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile="RESEARCH")
profile_info = analyzer.composite_calculator.get_profile_info()

config_data = {
    "Detected Architecture": detected_arch,
    "Available Profiles": ", ".join(list(csc.PROFILE_WEIGHTS.keys())),
    "Selected Profile": profile_info['profile'],
    "Profile Description": profile_info['description'],
    "Profile Weights": str(profile_info['weights'])
}

# 3. convert the dictionary items into a list of [key, value] pairs.
config_df = pd.DataFrame(
    list(config_data.items()),
    columns=['Configuration Parameter', 'Value']
)

# 4. Setting the parameter name as the index makes the table look cleaner.
config_df.set_index('Configuration Parameter', inplace=True)
print(config_df) # or display(config_df)

########################################################################
# --- Enhanced Analysis using Benchmark Suite for proper calibration ---
print("\n"*2 + '#'*150 + "\n")
print("\t"*8 + "\033[1mENHANCED ALGORITHM ANALYSIS: FULL SUITE SUMMARY\033[0m")


# 1. Run the entire suite using the 'benchmark_suite' method.
# This method first collects all raw results AND THEN updates the
# reference values within the analyzer object before calculating scores.
#
# IMPORTANT: The results inside benchmark_results are calculated AFTER
# the reference values have been calibrated on the suite itself.
algorithms_for_benchmark = list(ba.algorithms_collection.items())
benchmark_results = analyzer.benchmark_suite(algorithms_for_benchmark)

# 2. Extract the results dictionary and convert to a DataFrame
# The actual results are under the 'results' key.
results_data = list(benchmark_results['results'].values())
for i, name in enumerate(benchmark_results['results'].keys()):
    results_data[i]['Algorithm'] = name

results_df = pd.DataFrame(results_data)


# 3. Prepare the DataFrame for display (same as before)
display_columns = [
    'Algorithm',
    'CU',
    'EU',
    'CO2',
    '$',
    'CU_normalized',
    'EU_normalized',
    'CO2_normalized',
    '$_normalized',
    'SCORE_GRADE',
    'EFFICIENCY_RATING',
    'COMPOSITE_SCORE'
]
# Ensure all columns exist before trying to display them
# This handles cases where some results might not be generated
existing_display_columns = [col for col in display_columns if col in results_df.columns]
results_df = results_df[existing_display_columns]

# Sort the table by the composite score in descending order
results_df = results_df.sort_values(by='COMPOSITE_SCORE', ascending=False)

# Set the 'Algorithm' column as the table index
results_df.set_index('Algorithm', inplace=True)

# Set display options for float numbers
pd.options.display.float_format = '{:,.6f}'.format

# 4. Print the final, formatted table
print(results_df) # or display(results_df)

########################################################################
# --- Detailed Comparison of All Algorithms Against the Best Performer ---

print("\n"*2 + '#'*150 + "\n")
print("\t"*4 + "\033[1mDETAILED COMPARISON AGAINST THE BEST PERFORMING ALGORITHM\033[0m")

# 1. Identify the best algorithm to use as our baseline for comparison.
# We get this information from the statistics calculated by the benchmark_suite.
baseline_name = benchmark_results['statistics']['best_algorithm']
baseline_metrics = benchmark_results['results'][baseline_name]

print(f"\n[Comparison] Baseline Algorithm (Highest Score): '{baseline_name}'")

# 2. Prepare a list to hold the comparison data for each algorithm.
comparison_data = []

# Iterate through all algorithm results to compare them against the baseline.
for name, current_metrics in benchmark_results['results'].items():
    # Calculate the difference in composite score.
    score_diff = current_metrics['COMPOSITE_SCORE'] - baseline_metrics['COMPOSITE_SCORE']

    # Calculate the percentage change for each raw metric.
    # Formula: ((current - baseline) / baseline) * 100
    # A positive value means it's "more expensive" or "worse" than the baseline.
    comparison_row = {
        'Algorithm': name,
        'COMPOSITE_SCORE': current_metrics['COMPOSITE_SCORE'],
        'Score_vs_Best': score_diff, # Will be 0 for the best, negative for others.
    }

    for metric in ["CU", "EU", "CO2", "$"]:
        current_val = current_metrics[metric]
        baseline_val = baseline_metrics[metric]

        # Avoid division by zero, though unlikely with this data.
        if baseline_val > 0:
            percent_change = ((current_val - baseline_val) / baseline_val) * 100
        else:
            percent_change = float('inf')

        comparison_row[f'{metric}_%_vs_Best'] = percent_change

    comparison_data.append(comparison_row)

# 3. Create the comparison DataFrame.
comparison_df = pd.DataFrame(comparison_data)
comparison_df.sort_values(by='COMPOSITE_SCORE', ascending=False, inplace=True)
comparison_df.set_index('Algorithm', inplace=True)

# 4. Format the DataFrame for better readability using the .style attribute.
# This gives us more control over formatting individual columns.
formatted_comparison = comparison_df.style.format({
    'COMPOSITE_SCORE': '{:.2f}',
    'Score_vs_Best': '{:+.2f}', # Add a '+' sign for positive numbers (only the baseline will be 0)
    'CU_%_vs_Best': '{:+.2f}%',
    'EU_%_vs_Best': '{:+.2f}%',
    'CO2_%_vs_Best': '{:+.2f}%',
    '$_%_vs_Best': '{:+.2f}%',
}).background_gradient(
    cmap='Reds', # 'Reds' cmap will highlight larger (worse) percentages in red
    subset=['CU_%_vs_Best', 'EU_%_vs_Best', 'CO2_%_vs_Best', '$_%_vs_Best']
).bar(
    subset=['Score_vs_Best'], # Add bars to show the score difference visually
    align='zero',
    color=['#d65f5f', '#5fba7d'] # Red for negative, green for positive
)

# Display the final, styled table.
print(formatted_comparison) # or display(formatted_comparison)

########################################################################
# --- Final Summaries: Statistics, Profile Comparison, and Environment ---

print("\n"*2 + '#'*150 + "\n")
print("\t"*5 + "\033[1mFINAL SUMMARIES AND CONTEXTUAL ANALYSIS\033[0m")

# --- Block 1: Benchmark Suite Statistics ---
print("\n[Benchmark] Statistical Summary:")

# The statistics are already in a dictionary, perfect for a key-value table.
stats_data = benchmark_results['statistics']
stats_df = pd.DataFrame(
    list(stats_data.items()),
    columns=['Statistic', 'Value']
)
stats_df.set_index('Statistic', inplace=True)

# Format float values to 2 decimal places for readability
stats_df['Value'] = stats_df['Value'].apply(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
print(stats_df) # or display(stats_df)


# --- Block 2: Profile Comparison Analysis ---
print(f"\n[Profile Comparison] How a single algorithm's score changes based on the active profile: {detected_arch}")
data = []
for metric, refs in benchmark_results['updated_references'].items():
    data.append({
        'metric': metric,
        'min': f"{refs['min']:g}",
        'max': f"{refs['max']:g}",
        'typical': f"{refs['typical']:g}"
    })

df = pd.DataFrame(data)
print(df) # or display(df)

# We'll use a simple, consistent algorithm for this "what-if" analysis.
test_algorithm_func = ba.algorithms_collection['Constant_O(1)_Formula']

print(f"\n[Info] Using '{list(ba.algorithms_collection.keys())[0]}' as the test case (current environment).")

profile_comparison_results = []

# Iterate through all available profiles defined in PROFILE_WEIGHTS.
for profile_name in csc.PROFILE_WEIGHTS.keys():

    # 1. Create a temporary, clean analyzer for each profile.
    temp_analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile=profile_name)

    # 3. Now, analyze our single test function using the fully configured and calibrated analyzer.
    # The score will now be calculated relative to the *calibrated* min/max for that profile's logic.
    test_result = temp_analyzer.analyze_function(test_algorithm_func)

    profile_comparison_results.append({
        'Profile': profile_name,
        'Composite Score': test_result['COMPOSITE_SCORE'],
        'Grade': test_result['SCORE_GRADE']
    })

# Create and display the DataFrame for the profile comparison.
profile_df = pd.DataFrame(profile_comparison_results)
profile_df.sort_values(by='Composite Score', ascending=False, inplace=True)
profile_df.set_index('Profile', inplace=True)

# Display the final table.
print(profile_df.style.format({'Composite Score': '{:.2f}'}))

# or display(profile_df.style.format({'Composite Score': '{:.2f}'}))



# --- Block 3: Environmental Impact Analysis ---
# Fetch the carbon intensity value.
carbon_intensity = analyzer.fetch_carbon_intensity()
print(f"\n[Environment] Current Carbon Intensity: {carbon_intensity:g} kgCO2/kWh")


##############################################################
####            REPORTING AND SAVING RESULTS              ####
##############################################################

print("\n"*2 + '#'*150 + "\n")
print("\t"*8 + "\033[1mREPORTING AND SAVING RESULTS\033[0m")

# --- 1. Generate All Data and Visualizations (always happens) ---

# a) Generate the summary chart for all algorithms
print("\n[Chart] Generating overall benchmark summary chart...")
u.create_benchmark_summary_chart(benchmark_results, DEFAULT_REPORT_DIR)
print("  > Chart is displayed above.")

# --- Generate Comparison Charts for All Algorithms ---
print("\n[Chart] Generating comparison charts for each algorithm vs. the best...")

# 1. Identify the baseline algorithm (the best one).
best_algo_name = benchmark_results['statistics']['best_algorithm']
best_algo_results = benchmark_results['results'][best_algo_name]
print(f"  > Baseline for comparison is '{best_algo_name}'.")

# 2. IMPORTANT: Create the main reports directory BEFORE the loop.
# The REPORT_DIR variable comes from the main script context (e.g., "enhanced_reports").
os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)

# 3. Loop through all algorithms and generate a chart for each one.
for current_algo_name, current_algo_results in benchmark_results['results'].items():

    if current_algo_name == best_algo_name:
        continue

    safe_current_name = u.make_safe_filename(current_algo_name)
    safe_best_name = u.make_safe_filename(best_algo_name)
    chart_filename = f"comparison_{safe_current_name}_vs_{safe_best_name}.png"

    # Construct the full, final path for the image file.
    output_filepath = os.path.join(DEFAULT_REPORT_DIR, chart_filename)

    print(f"  > Generating chart: {chart_filename}")

    # Call the updated chart function with the full file path.
    u.create_enhanced_comparison_chart(
        result1=best_algo_results,
        result2=current_algo_results,
        output_filepath=output_filepath, # Pass the full path
        names=(best_algo_name, current_algo_name)
    )

print("  > All comparison charts have been generated and displayed.")

# --- Generate and Display Comprehensive Summary Report ---

print("\n" + "="*110)
print("\t"*5 + "\033[1mCOMPREHENSIVE SUMMARY REPORT\033[0m")
print("="*110)

# 1. Prepare all the data components (no change here)
best_algo_name = benchmark_results['statistics']['best_algorithm']
worst_algo_name = benchmark_results['statistics']['worst_algorithm']
best_algo_results = benchmark_results['results'][best_algo_name]
worst_algo_results = benchmark_results['results'][worst_algo_name]

# 2. Create the summary_report dictionary as before
summary_report = {
    "analysis_timestamp": platform.platform(),
    "architecture": detected_arch,
    "profile_used": profile_info,
    "benchmark_summary_stats": benchmark_results["statistics"],
    "best_algorithm_details": best_algo_results,
    "worst_algorithm_details": worst_algo_results,
    "recommendations": u.generate_recommendations(benchmark_results, profile_info)
}

# 3. Flatten the complex dictionary into a list of [key, value] pairs for the DataFrame
report_data_list = []

# a) Add top-level info
report_data_list.append(['Analysis Timestamp', summary_report['analysis_timestamp']])
report_data_list.append(['Architecture', summary_report['architecture']])

# b) Unroll the 'profile_used' dictionary
for key, value in summary_report['profile_used'].items():
    # Make keys more descriptive, e.g., "Profile - Description"
    descriptive_key = f"Profile - {key.replace('_', ' ').capitalize()}"
    report_data_list.append([descriptive_key, str(value)])

# c) Unroll the 'benchmark_summary_stats' dictionary
for key, value in summary_report['benchmark_summary_stats'].items():
    descriptive_key = f"Stat - {key.replace('_', ' ').title()}"
    # Format numbers for better display
    formatted_value = f"{value:.2f}" if isinstance(value, float) else value
    report_data_list.append([descriptive_key, formatted_value])

# d) Unroll the 'recommendations' list
for i, rec_text in enumerate(summary_report['recommendations']):
    report_data_list.append([f'Recommendation #{i+1}', rec_text])

# e) Unroll details for the BEST algorithm
for key, value in summary_report['best_algorithm_details'].items():
    if key == 'Algorithm': continue # Skip redundant name
    descriptive_key = f"Best Algo ({best_algo_name}) - {key}"
    formatted_value = f"{value:g}" if isinstance(value, float) else value
    report_data_list.append([descriptive_key, formatted_value])

# f) Unroll details for the WORST algorithm
for key, value in summary_report['worst_algorithm_details'].items():
    if key == 'Algorithm': continue # Skip redundant name
    descriptive_key = f"Worst Algo ({worst_algo_name}) - {key}"
    formatted_value = f"{value:g}" if isinstance(value, float) else value
    report_data_list.append([descriptive_key, formatted_value])


# 4. Create and display the DataFrame
summary_df = pd.DataFrame(report_data_list, columns=['Metric', 'Value'])
summary_df.set_index('Metric', inplace=True)
print(summary_df) # or display(summary_df)


# --- 2. Handle File Saving Operations based on SAVING_FLAG ---

if not SAVING_FLAG:
    # If not saving, just print a confirmation that visuals were displayed.
    print("\n[Info] SAVING_FLAG is set to False. All results and charts are displayed above but not saved to disk.")

else:
    # If saving is enabled, proceed with all file I/O operations.
    print(f"\n[Info] SAVING_FLAG is True. Saving all reports to directory: '{DEFAULT_REPORT_DIR}'")

    # The directory is already created by the charting functions, so we just confirm.
    os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)

    # a) Save main DataFrames as CSV files
    results_df.to_csv(os.path.join(DEFAULT_REPORT_DIR, "01_full_benchmark_summary.csv"))
    comparison_df.to_csv(os.path.join(DEFAULT_REPORT_DIR, "02_comparison_vs_best.csv"))
    print("  > Saved main DataFrames to CSV.")

    # b) Save raw results dictionaries as JSON files
    with open(os.path.join(DEFAULT_REPORT_DIR, "03_benchmark_suite_raw_data.json"), "w", encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=4)
    print("  > Saved raw benchmark data to JSON.")

    # c) Save the comprehensive summary report
    print("\n[Info] Saving comprehensive summary report to JSON file...")
    try:
        with open(os.path.join(DEFAULT_REPORT_DIR, "comprehensive_summary_report.json"), "w", encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        print("  > Report saved successfully.")
    except Exception as e:
        print(f"  > Error saving summary report: {e}")

    # d) Create a ZIP archive of all generated reports
    print("  > Creating ZIP archive of all reports...")
    try:
        archive_path = shutil.make_archive('enhanced_reports', 'gztar', root_dir=DEFAULT_REPORT_DIR)
        print(f"  > Successfully created archive: '{archive_path}'")
    except Exception as e:
        print(f"  > Error creating archive: {e}")

# --- Final Confirmation Message ---
print("\n" + "="*80)
print("\033[1mANALYSIS AND REPORTING COMPLETE\033[0m")
if SAVING_FLAG:
    print(f"All reports have been saved in the '{DEFAULT_REPORT_DIR}' directory and compressed into '{archive_path}'.")
else:
    print("Analysis results were displayed above. No files were saved.")
print("="*80)

