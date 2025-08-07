# analysis_external_repos.py

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
        
# if you need to delete all of them
# for repo_url in REPO_URLS:
#  local_repo_path = repo_url.split('/')[-1].replace('.git', '')
#  shutil.rmtree('./' + local_repo_path)

###########################################
####            plotting               ####
###########################################

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
