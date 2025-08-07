# analysis_external_repo_one.py

import platform
import os

import utils as u


###########################################
####            main body              ####
###########################################

print("\n"*2 + '#'*150 + "\n")
print("\t"*6 + "\033[1mCOMPREHENSIVE REPOSITORY ANALYSIS\033[0m")

REPO_URL = "https://github.com/s-kav/ds_tools.git"      # you can change this
LOCAL_REPO_PATH = "ds_tools"                            # you can change this
detected_arch = platform.machine().lower()

if not os.path.isdir(LOCAL_REPO_PATH):
    print(f"[Info] Repository '{LOCAL_REPO_PATH}' not found locally. Cloning from {REPO_URL}...")
    !git clone {REPO_URL}
    print("[Info] Cloning complete.")
else:
    print(f"[Info] Repository '{LOCAL_REPO_PATH}' already exists locally.")

# The 'detected_arch' variable must be defined from your first initialization block
repository_df = u.analyze_repository(
    repo_path=LOCAL_REPO_PATH,
    detected_arch=detected_arch, # Pass the architecture here
    verbose=False
)
repository_df.set_index('PROFILE NAME', inplace=True)

if not repository_df.empty:
    print(f"\n[Analysis] Aggregated assessment for repository: '{LOCAL_REPO_PATH}'")
    display(repository_df.style.format({
        'COMPOSITE_SCORE': '{:.2f}',
        'CU': '{:,.0f}',
        'EU': '{:,.4f}',
        'CO2': '{:,.4f}',
        '$': '{:,.4f}',
    }))

###########################################
####            plotting               ####
###########################################

if not repository_df.empty:
    print("\n[Chart] Generating custom performance scatter plot...")
    
    scatterplot_filepath = os.path.join(u.DEFAULT_REPORT_DIR, "repository_performance_scatterplot.png")
    
    # Call the new function. You can change the y-axis metric here if you want.
    # For example: y_axis_metric='EU'
    u.create_file_performance_scatterplot(
        data=repository_df,
        output_filepath=scatterplot_filepath,
        y_axis_metric='CU' # This can be changed to 'EU', 'CO2', or '$'        
    )
