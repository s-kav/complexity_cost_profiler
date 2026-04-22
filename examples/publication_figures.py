#!/usr/bin/env python3
# complexity_cost_profiler/examples/publication_figures.py
"""
Publication-quality figures for mixed-precision HPC cost analysis.
 
Generates five figures suitable for a scientific journal article:
  Fig. 1 — Pareto front: speedup vs. accuracy loss
  Fig. 2 — Relative cost reduction per metric (grouped bar chart)
  Fig. 3 — Composite efficiency score by precision format
  Fig. 4 — HPC library benchmark: elapsed time vs. problem size
  Fig. 5 — Task–precision compatibility heatmap
 
Run from the repository root:
    PYTHONPATH=src python examples/publication_figures.py
 
Output: figures/fig{1..5}.png  (300 dpi PNG for preview)
"""
 
import os
import sys
import math
 
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
 
from precision_accuracy_model import PrecisionAccuracyModel, SUPPORTED_PRECISIONS
from enhanced_cost_analyzer    import EnhancedCostAnalyzer
from hpc_library_integration   import HPCLibraryProfiler
 
# ---------------------------------------------------------------------------
# Global style — mimics IEEE / Elsevier single-column figure conventions
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})
 
PREC_COLORS = {
    "FP64": "#1a5276",
    "FP32": "#1f618d",
    "BF16": "#f39c12",
    "FP16": "#e67e22",
    "INT8": "#c0392b",
}
PREC_MARKERS = {
    "FP64": "o", "FP32": "s", "BF16": "^", "FP16": "D", "INT8": "X",
}
 
OUT_DIR = os.path.join(_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
 
 
def _save(fig: plt.Figure, name: str) -> None:
    # Only PNG, no PDF (as requested)
    for ext in ("png",):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
 
 
# ===========================================================================
# Fig. 1 — Pareto front: throughput speedup vs. accuracy loss factor
# ===========================================================================
 
def fig1_pareto_front() -> None:
    print("Generating Fig. 1 — Pareto front …")
    model = PrecisionAccuracyModel()
    base  = {"CU": 200.0, "EU": 2.0, "CO2": 0.000154, "$": 0.0008}
    table = model.build_tradeoff_table(base, base_precision="FP64")
 
    # Increased figure size
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
 
    for row in table:
        p     = row["Precision"]
        x     = row["Speedup_vs_base"]
        y     = row["Accuracy_loss_factor"]
        color = PREC_COLORS[p]
        mark  = PREC_MARKERS[p]
        # Smaller markers
        ax.scatter(x, y, s=40, color=color, marker=mark, zorder=5)
        # No inline labels (only legend)
 
    # Linear X scale (removed log)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Throughput Speedup Relative to FP64", labelpad=4)
    ax.set_ylabel("Accuracy Loss Factor\n"
                  r"($\varepsilon_{\rm target}\,/\,\varepsilon_{\rm FP64}$)", labelpad=4)
    ax.set_title("Precision Format Trade-off:\nSpeedup vs. Accuracy Loss")
 
    # Legend at lower right
    legend_handles = [
        Line2D([0], [0], marker=PREC_MARKERS[p], color="w",
               markerfacecolor=PREC_COLORS[p], markersize=7, label=p)
        for p in SUPPORTED_PRECISIONS
    ]
    ax.legend(handles=legend_handles, loc="lower right", title="Format",
              title_fontsize=8, borderpad=0.5, frameon=True)
 
    fig.tight_layout()
    _save(fig, "fig1_pareto_front")
 
 
# ===========================================================================
# Fig. 2 — Relative cost reduction (CU / EU / CO2 / $)  vs. FP64
# ===========================================================================
 
def fig2_cost_reduction() -> None:
    print("Generating Fig. 2 — Cost reduction grouped bar chart …")
    model = PrecisionAccuracyModel()
    base  = {"CU": 200.0, "EU": 2.0, "CO2": 0.000154, "$": 0.0008}
    table = model.build_tradeoff_table(base, base_precision="FP64")
 
    # Remove FP64 from the table (only other precisions)
    table = [row for row in table if row["Precision"] != "FP64"]
 
    metrics   = ["Energy_saved_%", "CO2_saved_%", "Cost_saved_%"]
    m_labels  = ["Energy\nsaving (%)", "CO₂\nsaving (%)", "Cost\nsaving (%)"]
    precs     = [r["Precision"] for r in table]
    n_prec    = len(precs)
    n_m       = len(metrics)
    width     = 0.2
    x         = np.arange(n_m)
 
    # Increased figure size
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
 
    for i, row in enumerate(table):
        p      = row["Precision"]
        vals   = [row[m] for m in metrics]
        offset = (i - n_prec / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               color=PREC_COLORS[p], label=p,
               edgecolor="white", linewidth=0.4)
 
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.set_ylabel("Reduction Relative to FP64 (%)")
    ax.set_title("Relative Resource Savings by Precision Format\n"
                 "(compared to FP64 baseline)")
    ax.set_ylim(0, 105)
    # Y ticks every 10, without % symbol (already in ylabel)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    # Legend without frame, placed below, but here we keep it as is but remove frame
    ax.legend(title="Format", ncol=len(precs), loc="upper center",
              bbox_to_anchor=(0.5, -0.15), frameon=False)
 
    fig.tight_layout()
    _save(fig, "fig2_cost_reduction")
 
 
# ===========================================================================
# Fig. 3 — Composite efficiency score by precision (MIXED_PRECISION profile)
# ===========================================================================
 
def fig3_composite_score() -> None:
    print("Generating Fig. 3 — Composite efficiency scores …")
 
    def _sample_fn():
        n = 64
        A = [[float(i + j) for j in range(n)] for i in range(n)]
        b = [float(i) for i in range(n)]
        result = []
        for row in A:
            s = 0.0
            for a_ij, b_j in zip(row, b):
                s += a_ij * b_j
            result.append(s)
        return result
 
    analyzer   = EnhancedCostAnalyzer(arch="x86_64", profile="MIXED_PRECISION")
    comparison = analyzer.compare_precisions(_sample_fn, precisions=SUPPORTED_PRECISIONS,
                                             include_composite=True)
    scores     = {p: comparison["by_precision"][p].get("COMPOSITE_SCORE", 0)
                  for p in SUPPORTED_PRECISIONS}
    grades     = {p: comparison["by_precision"][p].get("SCORE_GRADE", "")
                  for p in SUPPORTED_PRECISIONS}
 
    # Increased figure size
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
 
    for i, p in enumerate(SUPPORTED_PRECISIONS):
        bar = ax.bar(i, scores[p], color=PREC_COLORS[p],
                     edgecolor="white", linewidth=0.5, width=0.6)
        # Add numeric value on top of bar
        ax.text(i, scores[p] + 1.5, f"{scores[p]:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=PREC_COLORS[p])
        # Grade text (already present)
        ax.text(i, scores[p] - 5, grades[p],
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="white")
 
    ax.set_xticks(range(len(SUPPORTED_PRECISIONS)))
    ax.set_xticklabels(SUPPORTED_PRECISIONS)
    ax.set_ylabel("Composite Efficiency Score (0–100)")
    ax.set_title("Composite Efficiency Score by Precision Format\n"
                 "(MIXED_PRECISION profile: dense matrix–vector product, $n=64$)")
    ax.set_ylim(0, 105)
    # Y ticks every 10
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
 
    # Reference lines
    for level, label in [(90, "A+"), (70, "B"), (50, "C−")]:
        ax.axhline(level, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.text(len(SUPPORTED_PRECISIONS) - 0.45, level + 0.5, label,
                fontsize=7.5, color="gray", va="bottom")
 
    fig.tight_layout()
    _save(fig, "fig3_composite_score")
 
 
# ===========================================================================
# Fig. 4 — HPC library benchmark: elapsed time vs. problem size
# ===========================================================================
 
def fig4_library_benchmark() -> None:
    print("Generating Fig. 4 — Library benchmark (this may take ~30 s) …")
 
    sizes     = [32, 64, 128, 256]
    precisions = ["FP64", "FP32"]
    profiler  = HPCLibraryProfiler(sizes=sizes, precisions=precisions,
                                   repeats=5, composite_profile="MIXED_PRECISION")
    results   = profiler.profile_all(operations=["matrix_multiply", "fft"])
 
    ops        = ["matrix_multiply", "fft"]
    op_labels  = ["Dense GEMM ($C = AB$)", "1-D FFT"]
    lib_styles = {"numpy": "-", "scipy": "--"}
    prec_alpha = {"FP64": 1.0, "FP32": 0.6}
 
    # Slightly increased figure size
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)
 
    # Collect all legend handles/labels once
    legend_handles = []
    legend_labels = []
 
    for ax, op, op_label in zip(axes, ops, op_labels):
        ax.set_title(op_label)
        ax.set_xlabel("Problem Size $n$")
        ax.set_ylabel("Elapsed Time (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
 
        plotted = set()
        for lib in ("numpy", "scipy"):
            for prec in precisions:
                pts = sorted(
                    [r for r in results
                     if r.get("operation") == op
                     and r.get("library") == lib
                     and r.get("precision") == prec
                     and not math.isnan(r.get("elapsed_ms", float("nan")))],
                    key=lambda r: r["size"]
                )
                if not pts:
                    continue
                xs = [r["size"]       for r in pts]
                ys = [r["elapsed_ms"] for r in pts]
                color  = PREC_COLORS[prec]
                ls     = lib_styles.get(lib, "-")
                alpha  = prec_alpha.get(prec, 1.0)
                lbl    = f"{lib}/{prec}"
                line, = ax.plot(xs, ys, ls=ls, marker=PREC_MARKERS[prec],
                                color=color, alpha=alpha, markersize=5,
                                linewidth=1.2, label=lbl)
                if lbl not in plotted:
                    legend_handles.append(line)
                    legend_labels.append(lbl)
                    plotted.add(lbl)
        # No individual legend per subplot
        ax.legend().remove()
 
    # Single legend for the whole figure
    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=len(legend_labels), frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
 
    fig.suptitle("HPC Library Benchmark: Elapsed Time vs. Problem Size\n"
                 "(CPU node, Intel Xeon Platinum 8380)", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_library_benchmark")
 
 
# ===========================================================================
# Fig. 5 — Task–precision compatibility heatmap
# ===========================================================================
 
def fig5_task_heatmap() -> None:
    print("Generating Fig. 5 — Task–precision compatibility heatmap …")
 
    model = PrecisionAccuracyModel()
 
    tasks = [
        ("climate_simulation",    "Climate / CFD"),
        ("molecular_dynamics",    "Molecular Dynamics"),
        ("financial_modeling",    "Financial Modeling"),
        ("linear_algebra_direct", "Direct Linear Algebra"),
        ("iterative_solver",      "Iterative Solver"),
        ("general_scientific",    "General Scientific"),
        ("signal_processing_fft", "Signal Processing / FFT"),
        ("deep_learning_training","DL Training"),
        ("dl_inference",          "DL Inference"),
        ("computer_vision",       "Computer Vision"),
        ("nlp_llm",               "NLP / LLM"),
        ("recommendation_systems","Recommendation Systems"),
    ]
 
    prec_idx = {p: i for i, p in enumerate(SUPPORTED_PRECISIONS)}
    n_tasks  = len(tasks)
    n_prec   = len(SUPPORTED_PRECISIONS)
 
    # Matrix values:  2 = minimum required (best),  1 = usable,  0 = unusable
    matrix = np.zeros((n_tasks, n_prec))
    for ti, (task_key, _) in enumerate(tasks):
        req       = model.task_requirements.get(task_key, {})
        min_prec  = req.get("min_precision", "FP64")
        min_idx   = prec_idx.get(min_prec, 0)
        for pi in range(n_prec):
            if pi == min_idx:
                matrix[ti, pi] = 2      # minimum required precision
            elif pi > min_idx:
                matrix[ti, pi] = 1      # lower precision — also usable
            else:
                matrix[ti, pi] = 0      # too low precision
 
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
 
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap  = ListedColormap(["#f8d7da", "#d4edda", "#155724"])
    norm  = BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
    im    = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
 
    ax.set_xticks(range(n_prec))
    ax.set_xticklabels(SUPPORTED_PRECISIONS, fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([label for _, label in tasks], fontsize=8.5)
    ax.set_xlabel("Floating-Point Precision Format")
    ax.set_title("Task–Precision Compatibility Matrix\n"
                 "(IEEE 754 accuracy requirements)")
 
    # Cell annotations
    symbols = {0: "N", 1: "Y", 2: "R"}   # R=required, Y=viable, N=not suitable
    for ti in range(n_tasks):
        for pi in range(n_prec):
            val = int(matrix[ti, pi])
            col = "white" if val == 2 else ("#aaa" if val == 0 else "#155724")
            ax.text(pi, ti, symbols[val], ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold",
                    fontstyle="normal")
 
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#155724", label="R  Minimum required precision"),
        Patch(facecolor="#d4edda", label="Y  Lower precision — viable"),
        Patch(facecolor="#f8d7da", label="N  Insufficient accuracy"),
    ]
    
    # Legend without frame, placed lower
    ax.legend(handles=legend_els, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8,
              frameon=False)
 
    fig.tight_layout()
    _save(fig, "fig5_task_heatmap")
 
 
# ===========================================================================
# Main
# ===========================================================================
 
if __name__ == "__main__":
    print("complexity_cost_profiler — Publication Figure Generator")
    print(f"Output directory: {OUT_DIR}\n")
 
    fig1_pareto_front()
    fig2_cost_reduction()
    fig3_composite_score()
    fig4_library_benchmark()
    fig5_task_heatmap()
 
    print("\nAll figures saved.")