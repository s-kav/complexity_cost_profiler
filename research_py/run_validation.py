# run_validation.py

import pandas as pd
from IPython.display import display, Image

# Import our project modules
import utils as u
import model_validator as mv

# --- Main Validation Execution Block ---
if __name__ == "__main__":
    print("#" * 80 + "\n### MODEL VALIDATION & ACCURACY REPORT ###\n" + "#" * 80)
    
    # 1. Generate the core dataset
    validation_df = mv.create_validation_dataset()
    
    # 2. Augment it with predictions from baseline models
    validation_df = mv.run_baseline_models(validation_df)
    
    print("\n--- Generated Validation Data (with all model predictions) ---")
    display(validation_df)

    # 3. Define the models to compare
    models_to_evaluate = {
        "Our Model": "predicted_cu",
        "Baseline B1 (Uniform Cost)": "b1_predicted_cu",
        "Baseline B2 (I/O Penalized)": "b2_predicted_cu",
    }
    
    error_results = []
    
    # 4. Calculate errors and generate plots for each model
    print("\n--- Calculating Errors and Generating Plots ---")
    for model_name, pred_col in models_to_evaluate.items():
        # Calculate error metrics
        errors = mv.calculate_error_metrics(validation_df, pred_col, 'measured_time_s')
        errors['model'] = model_name
        error_results.append(errors)
        
        # Generate plot
        safe_name = u.make_safe_filename(model_name)
        chart_path = f"{u.DEFAULT_REPORT_DIR}/validation_{safe_name}.png"
        plot_title = f"Validation: {model_name} vs. Measured Time"
        
        mv.plot_prediction_vs_measured(validation_df, pred_col, 'measured_time_s', plot_title, chart_path)
        print(f"Generated plot for '{model_name}': {chart_path}")
        display(Image(filename=chart_path))

    # 5. Display the final error comparison table
    error_df = pd.DataFrame(error_results).set_index('model')
    print("\n--- Model Accuracy Comparison ---")
    display(error_df.style.format({
        'MAE': '{:.4f}',
        'MAPE': '{:.2%}',
        'Spearman_ρ': '{:.3f}',
    }))

    # 6. Generate and print the final English summary
    print("\n--- Accuracy Summary ---")
    summary_text = mv.generate_accuracy_summary(error_df)
    print(summary_text)

    print("\n" + "#" * 80 + "\n### VALIDATION COMPLETE ###\n" + "#" * 80)