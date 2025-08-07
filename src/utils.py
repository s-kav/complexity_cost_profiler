# utils.py

import os
import csv
import json
import re
import zipfile
import glob
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from enhanced_cost_analyzer import EnhancedCostAnalyzer
from composite_score_calculator import CompositeScoreCalculator, PROFILE_WEIGHTS

# This constant now lives with the functions that use it.
DEFAULT_REPORT_DIR = "./enhanced_reports"


def save_enhanced_csv(data: Dict[str, Any], filename: str, report_dir: str = DEFAULT_REPORT_DIR) -> None:
    """Saves enhanced cost data to a CSV file."""
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, filename)
    with open(filepath, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value", "Description"])
        
        metrics_to_write = {
            "CU": "Raw CU value",
            "EU": "Raw EU value",
            "CO2": "Raw CO2 value",
            "$": "Raw $ value",
            "CU_normalized": "Normalized CU score (0-100)",
            "EU_normalized": "Normalized EU score (0-100)",
            "CO2_normalized": "Normalized CO2 score (0-100)",
            "$_normalized": "Normalized $ score (0-100)",
            "COMPOSITE_SCORE": "Unified composite score (0-100)",
            "SCORE_GRADE": "Letter grade rating",
            "EFFICIENCY_RATING": "Efficiency rating"
        }
        for key, desc in metrics_to_write.items():
            if key in data:
                writer.writerow([key, data[key], desc])


def generate_recommendations(benchmark_results: Dict[str, Any], profile_info: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generates algorithm optimization recommendations based on benchmark results."""
    recommendations = []
    stats = benchmark_results["statistics"]
    results = benchmark_results["results"]

    if profile_info:
        recommendations.append(f"Analysis performed using '{profile_info['profile']}' profile: {profile_info['description']}")

    best_alg = stats["best_algorithm"]
    recommendations.append(f"Use '{best_alg}' for optimal performance with composite score of {results[best_alg]['COMPOSITE_SCORE']:.1f}")

    worst_alg = stats["worst_algorithm"]
    recommendations.append(f"Avoid '{worst_alg}' due to poor performance with composite score of {results[worst_alg]['COMPOSITE_SCORE']:.1f}")

    if stats["score_range"] > 20:
        recommendations.append("Large performance variation detected - consider algorithm selection based on specific use case requirements.")

    return recommendations


def create_repository_comparison_chart(repo_data: Dict[str, pd.DataFrame], output_filepath: str, profile_to_plot: str = "TOTAL") -> None:
    """Creates a comparison chart for multiple repositories based on a specific profile row."""
    repo_names, composite_scores = [], []
    raw_metrics_data = {'CU': [], 'EU': [], 'CO2': [], '$': []}
    metrics_to_plot = list(raw_metrics_data.keys())

    for name, df in repo_data.items():
        if profile_to_plot in df.index:
            repo_names.append(name)
            row = df.loc[profile_to_plot]
            composite_scores.append(row['COMPOSITE_SCORE'])
            for metric in metrics_to_plot:
                raw_metrics_data[metric].append(row[metric])
    
    if not repo_names:
        print(f"[Chart Error] No data found for profile '{profile_to_plot}' in any repository.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f'Repository Performance Comparison (Profile: {profile_to_plot})', fontsize=16)

    # Subplot 1: Composite Score
    x_pos_ax1 = np.arange(len(repo_names))
    ax1.bar(x_pos_ax1, composite_scores, color='skyblue', alpha=0.8)
    ax1.set_title('Composite Score Comparison')
    ax1.set_ylabel('Score (Higher is Better)')
    ax1.set_xticks(x_pos_ax1)
    ax1.set_xticklabels(repo_names, rotation=45, ha='right')
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    for i, score in enumerate(composite_scores):
        ax1.text(i, score, f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    if composite_scores: ax1.set_ylim(0, max(composite_scores) * 1.15)

    # Subplot 2: Raw Costs
    x_pos_ax2 = np.arange(len(repo_names))
    n_metrics, total_bar_width = len(metrics_to_plot), 0.8
    bar_width = total_bar_width / n_metrics
    
    for i, metric in enumerate(metrics_to_plot):
        offset = -total_bar_width / 2 + i * bar_width + bar_width / 2
        ax2.bar(x_pos_ax2 + offset, raw_metrics_data[metric], bar_width, label=metric)

    ax2.set_title('Aggregated Raw Costs')
    ax2.set_ylabel('Cost Values (Log Scale, Lower is Better)')
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos_ax2)
    ax2.set_xticklabels(repo_names, rotation=45, ha='right')
    ax2.legend(title="Metrics")
    ax2.grid(True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xticks(x_pos_ax2 - total_bar_width/2, minor=True)
    ax2.grid(True, axis='x', which='minor', linestyle='-', linewidth=0.7, color='black', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_safe_filename(name: str) -> str:
    """Converts a string into a safe filename by removing special characters."""
    name = name.replace(' ', '_').replace('^', '').replace('(', '').replace(')', '')
    safe_name = re.sub(r'(?u)[^-\w.]', '', name)
    return safe_name


def create_file_performance_scatterplot(
    data: pd.DataFrame, 
    output_filepath: str,
    y_axis_metric: str = 'CU'
) -> None:
    """
    Creates a custom 2D scatter plot for file performance analysis, limited to the top 20 files.

    Args:
        data (pd.DataFrame): The DataFrame with analysis results.
        output_filepath (str): The full path where the PNG file will be saved.
        y_axis_metric (str): The metric to use for the Y-axis.
    """
    if data.empty:
        print("[Chart Info] Cannot generate scatter plot: The input DataFrame is empty.")
        return
        
    required_cols = ['COMPOSITE_SCORE', 'SCORE_GRADE', 'File Type', y_axis_metric]
    if not all(col in data.columns for col in required_cols):
        print(f"[Chart Error] Input data is missing one of the required columns: {required_cols}")
        return

    plot_df = data.nlargest(20, 'COMPOSITE_SCORE').copy()
    print(f"[Chart Info] Displaying top {len(plot_df)} files sorted by COMPOSITE_SCORE.")

    plt.figure(figsize=(12, 6))

    # Define lists of distinct markers and colors to cycle through
    markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>', 'p', 'h', '+', 'd', 'H']
    # Use a colormap to get a variety of distinct colors
    cmap = plt.get_cmap('tab20')
  
    legend_handles = []

    for i, (index, row) in enumerate(plot_df.iterrows()):
        marker = markers[i % len(markers)]
        color = cmap(i / len(plot_df))
        x_val = row['COMPOSITE_SCORE']
        y_val = row[y_axis_metric]
        
        plt.scatter(x_val, y_val, marker=marker, s=120, label=index, color=color) # CHANGED: Apply color

        legend_label = f"{index} ({row['File Type']})"
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=marker, linestyle='None', # CHANGED: Use point's color
                          markersize=10, label=legend_label)
        )
        
        plt.text(x_val, y_val, f" {row['SCORE_GRADE']}", 
                 verticalalignment='bottom', ha='left', fontsize=9, color='darkred',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')) # Added background to text

    # --- Final Touches ---
    plt.title(f'File Performance Analysis: Score vs. {y_axis_metric} Cost', fontsize=16)
    plt.xlabel('Composite Score (Higher is Better)', fontsize=12)
    plt.ylabel(f'{y_axis_metric} Cost (Lower is Better)', fontsize=12)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # CHANGED: Place the legend inside the plot. 'best' tries to find the least obstructive location.
    plt.legend(handles=legend_handles, title='Files', loc='best', fontsize='small')
    
    # Use the standard tight_layout
    plt.tight_layout()

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def analyze_repository(
    repo_path: str,
    detected_arch: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Performs a self-contained, comprehensive analysis of a code repository.
    This version correctly calculates the TOTAL row based on the average of profile scores.

    Args:
        repo_path (str): The path to the repository directory to analyze.
        detected_arch (str): The architecture string for the cost model.
        verbose (bool): If True, prints progress messages.

    Returns:
        pd.DataFrame: A summary DataFrame showing aggregated costs under different profiles.
    """
    if not os.path.isdir(repo_path):
        if verbose:
            print(f"[Error] Repository path not found: {repo_path}")
        return pd.DataFrame()

    file_types_to_analyze = {
        'Python': {'extension': 'py', 'analysis_key': 'analyze_py_file'},
        'LLVM IR': {'extension': 'll', 'analysis_key': 'analyze_llvm_ir'},
        'PTX GPU': {'extension': 'ptx', 'analysis_key': 'analyze_ptx'}
    }

    if verbose:
        print("\n--- Step 1 of 2: Collecting and aggregating raw performance data... ---")
    raw_analyzer = EnhancedCostAnalyzer(arch=detected_arch)
    aggregated_raw_metrics = {}

    for file_type_name, config in file_types_to_analyze.items():
        search_pattern = os.path.join(repo_path, f"**/*.{config['extension']}")
        found_files = glob.glob(search_pattern, recursive=True)
        if not found_files: continue

        type_aggregator = {'CU': 0, 'EU': 0, 'CO2': 0, '$': 0, 'function_count': 0, 'file_count': len(found_files)}
        analysis_func = getattr(raw_analyzer, config['analysis_key'])

        for file_path in found_files:
            # Вызываем функцию анализа в зависимости от ее типа
            if config['analysis_key'] == 'analyze_py_file':
                # analyze_py_file требует флаг verbose и возвращает список
                file_results = analysis_func(file_path, verbose=verbose)
            else:
                # Другие функции возвращают один словарь, который мы оборачиваем в список
                file_results = [analysis_func(file_path)]

            for result in file_results:
                if not result: continue
                for metric in ['CU', 'EU', 'CO2', '$']:
                    type_aggregator[metric] += result.get(metric, 0)
                func_name = result.get("Function Name", "")
                if func_name and not func_name.startswith('['):
                    type_aggregator['function_count'] += 1

        if type_aggregator['CU'] > 0:
            aggregated_raw_metrics[file_type_name] = type_aggregator

    if not aggregated_raw_metrics:
        if verbose:
            print("No analyzable content with non-zero cost found in the repository.")
        return pd.DataFrame()

    if verbose:
        print("\n--- Step 2 of 2: Applying each profile to the aggregated data... ---")
    final_results_list = []

    for profile_name in PROFILE_WEIGHTS.keys():
        if verbose:
            print(f"  > Scoring with profile: [{profile_name}]")
        profile_analyzer = EnhancedCostAnalyzer(arch=detected_arch, profile=profile_name)

        for file_type_name, raw_data in aggregated_raw_metrics.items():
            scored_row = profile_analyzer.composite_calculator.calculate_composite_score(raw_data)
            scored_row['PROFILE NAME'] = profile_name
            scored_row['File Type'] = f"{file_type_name} ({raw_data.get('file_count', 0)})"
            scored_row['Function Name'] = raw_data.get('function_count', 0)
            final_results_list.append(scored_row)

    repo_df = pd.DataFrame(final_results_list)

    if not repo_df.empty:
        average_composite_score = repo_df['COMPOSITE_SCORE'].mean()
        total_metrics = repo_df.loc[repo_df['PROFILE NAME'] == 'DEFAULT', ['CU', 'EU', 'CO2', '$']].sum().to_dict()
        temp_calculator = CompositeScoreCalculator()
        average_grade = temp_calculator._get_score_grade(average_composite_score)
        total_file_count = sum(d.get('file_count', 0) for d in aggregated_raw_metrics.values())
        total_func_count = sum(d.get('function_count', 0) for d in aggregated_raw_metrics.values())

        total_row = {
            'PROFILE NAME': 'TOTAL',
            'File Type': f"All Files ({total_file_count})",
            'Function Name': total_func_count,
            'COMPOSITE_SCORE': average_composite_score,
            'SCORE_GRADE': average_grade,
            'CU': total_metrics.get('CU', 0),
            'EU': total_metrics.get('EU', 0),
            'CO2': total_metrics.get('CO2', 0),
            '$': total_metrics.get('$', 0)
        }
        
        final_df = pd.concat([repo_df, pd.DataFrame([total_row])], ignore_index=True)
    else:
        final_df = repo_df

    final_columns = ['PROFILE NAME', 'File Type', 'Function Name', 'COMPOSITE_SCORE', 'SCORE_GRADE', 'CU', 'EU', 'CO2', '$']
    final_df = final_df.reindex(columns=final_columns)

    return final_df


def create_repository_comparison_chart(
    repo_data: Dict[str, pd.DataFrame],
    output_filepath: str,
    profile_to_plot: str = "TOTAL"
) -> None:
    """
    Creates a comparison chart for multiple repositories based on a specific profile row.

    Args:
        repo_data (Dict[str, pd.DataFrame]): A dictionary mapping repository names to their analysis DataFrames.
        output_filepath (str): The full path where the PNG file will be saved.
        profile_to_plot (str): The name of the profile row (e.g., 'TOTAL', 'RESEARCH') to use for plotting.
    """
    repo_names = []
    composite_scores = []
    raw_metrics_data = {'CU': [], 'EU': [], 'CO2': [], '$': []}
    metrics_to_plot = list(raw_metrics_data.keys())

    for name, df in repo_data.items():
        if profile_to_plot in df.index:
            repo_names.append(name)
            row = df.loc[profile_to_plot]
            composite_scores.append(row['COMPOSITE_SCORE'])
            for metric in metrics_to_plot:
                raw_metrics_data[metric].append(row[metric])
    
    if not repo_names:
        print(f"[Chart Error] No data found for profile '{profile_to_plot}' in any repository.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle(f'Repository Performance Comparison (Profile: {profile_to_plot})', fontsize=16)

    # --- Subplot 1: Composite Score (Higher is Better) ---
    x_pos_ax1 = np.arange(len(repo_names))
    ax1.bar(x_pos_ax1, composite_scores, color='skyblue', alpha=0.8)
    ax1.set_title('Composite Score Comparison (Total)')
    ax1.set_ylabel('Score (Higher is Better)')
    ax1.set_xticks(x_pos_ax1)
    ax1.set_xticklabels(repo_names, rotation=45, ha='right')
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7) # Added horizontal grid for ax1 too
    
    for i, score in enumerate(composite_scores):
        ax1.text(i, score, f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    if composite_scores:
        ax1.set_ylim(0, max(composite_scores) * 1.15)

    # --- Subplot 2: Raw Costs (Lower is Better) - Grouped Bar Chart ---
    x_pos_ax2 = np.arange(len(repo_names))
    n_metrics = len(metrics_to_plot)
    total_bar_width, bar_width = 0.8, 0.8 / n_metrics
    
    for i, metric in enumerate(metrics_to_plot):
        offset = -total_bar_width / 2 + i * bar_width + bar_width / 2
        ax2.bar(x_pos_ax2 + offset, raw_metrics_data[metric], bar_width, label=metric)

    ax2.set_title('Aggregated Raw Costs')
    ax2.set_ylabel('Cost Values (Log Scale, Lower is Better)')
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos_ax2)
    ax2.set_xticklabels(repo_names, rotation=45, ha='right')
    ax2.legend(title="Metrics")

    # --- Grid and Axis Configuration for ax2 ---
    # 1. Add horizontal grid lines
    ax2.grid(True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    # 2. Add vertical grid lines between groups
    ax2.set_xticks(x_pos_ax2 + total_bar_width/2 + (bar_width/2 if n_metrics > 1 else 0.1), minor=True) # Position minor ticks between groups
    ax2.grid(True, axis='x', which='minor', linestyle='-', linewidth=0.7, color='black', alpha=0.6) # Draw grid for minor ticks
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    # plt.show() # show() is not needed if we display the image file directly
    plt.close(fig) # Close the plot to free up memory


def create_benchmark_summary_chart(benchmark_results: Dict[str, Any], report_dir: str) -> None:
    """
    Creates a summary chart for the benchmark suite results.
    This function is now considered deprecated in favor of the more versatile
    create_repository_comparison_chart.
    """
    results = benchmark_results.get("results", {})
    if not results:
        print("[Chart Info] No results found in benchmark data to generate a chart.")
        return

    df = pd.DataFrame.from_dict(results, orient='index')
    df.sort_values(by="COMPOSITE_SCORE", ascending=False, inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Benchmark Suite Summary', fontsize=16)

    # --- Bar Chart: Composite Scores ---
    x_pos = np.arange(len(df.index))
    ax1.bar(x_pos, df['COMPOSITE_SCORE'], color='lightblue')
    ax1.set_title('Algorithm Composite Scores')
    ax1.set_ylabel('Score (Higher is Better)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df.index, rotation=45, ha='right') # <-- ИСПРАВЛЕННЫЙ МЕТОД
    ax1.set_ylim(0, 110)

    # Add text labels on bars
    for i, score in enumerate(df['COMPOSITE_SCORE']):
        ax1.text(i, score + 1, f'{score:.1f}', ha='center')

    # --- Pie Chart: Contribution to Total Cost (CU) ---
    # Highlight the algorithm with the highest computational cost (worst CU)
    worst_cu_alg = df['CU'].idxmax()
    explode = [0.1 if idx == worst_cu_alg else 0 for idx in df.index]

    ax2.pie(df['CU'], labels=df.index, autopct='%1.1f%%', startangle=90,
            explode=explode, shadow=True)
    ax2.set_title(f'Contribution to Total Computational Cost (CU)\n(Worst: {worst_cu_alg})')
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # --- Save the chart ---
    output_filepath = os.path.join(report_dir, "benchmark_suite_summary_chart.png")
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def create_enhanced_comparison_chart(
    result1: Dict[str, float],
    result2: Dict[str, float],
    output_filepath: str,  # Takes the full path to the output file
    names: Tuple[str, str] = ("Algorithm v1", "Algorithm v2")
) -> None:
    """
    Creates an enhanced comparison chart and saves it to a specified file path.

    Args:
        result1: Cost metrics for the baseline algorithm.
        result2: Cost metrics for the algorithm to compare.
        output_filepath: The full path where the PNG file will be saved.
        names: Names for the two algorithms for chart legends.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparison: "{names[1]}" vs. "{names[0]}"', fontsize=16)

    # Raw metrics comparison
    raw_metrics = ["CU", "EU", "CO2", "$"]
    values1_raw = [result1.get(m, 0) for m in raw_metrics]
    values2_raw = [result2.get(m, 0) for m in raw_metrics]
    x = np.arange(len(raw_metrics))
    width = 0.35
    ax1.bar(x - width/2, values1_raw, width, label=names[0], color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, values2_raw, width, label=names[1], color='lightcoral', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(raw_metrics)
    ax1.set_ylabel("Raw Values (Log Scale)")
    ax1.set_title("Raw Metrics Comparison")
    ax1.legend()
    ax1.set_yscale('log')

    # Normalized scores comparison
    normalized_metrics = [f"{m}_normalized" for m in raw_metrics]
    values1_norm = [result1.get(m, 50) for m in normalized_metrics]
    values2_norm = [result2.get(m, 50) for m in normalized_metrics]
    ax2.bar(x - width/2, values1_norm, width, label=names[0], color='skyblue', alpha=0.8)
    ax2.bar(x + width/2, values2_norm, width, label=names[1], color='lightcoral', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_normalized', '') for m in normalized_metrics])
    ax2.set_ylabel("Normalized Scores (0-100)")
    ax2.set_title("Normalized Scores Comparison")
    ax2.legend()
    ax2.set_ylim(0, 105)

    # Composite score comparison
    composite1 = result1.get("COMPOSITE_SCORE", 50)
    composite2 = result2.get("COMPOSITE_SCORE", 50)
    ax3.bar([names[0], names[1]], [composite1, composite2], color=['skyblue', 'lightcoral'], alpha=0.8)
    ax3.set_ylabel("Composite Score (0-100)")
    ax3.set_title("Composite Score Comparison")
    ax3.set_ylim(0, 105)
    for i, score in enumerate([composite1, composite2]):
        ax3.text(i, score, f"{score:.1f}", ha='center', va='bottom', fontsize=12)

    # Radar chart
    labels = np.array(raw_metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    ax4 = plt.subplot(2, 2, 4, polar=True)
    values1_radar = values1_norm + [values1_norm[0]]
    values2_radar = values2_norm + [values2_norm[0]]
    ax4.plot(angles, values1_radar, 'o-', linewidth=2, label=names[0], color='skyblue')
    ax4.fill(angles, values1_radar, 'skyblue', alpha=0.25)
    ax4.plot(angles, values2_radar, 'o-', linewidth=2, label=names[1], color='lightcoral')
    ax4.fill(angles, values2_radar, 'lightcoral', alpha=0.25)
    ax4.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax4.set_ylim(0, 100)
    ax4.set_title("Multi-dimensional Performance Radar", y=1.1)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # The function ONLY saves the file. It does not create directories.
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

