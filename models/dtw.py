#=====================================================================
#IEEE Transactions on Medical Imaging (T-MI)
#Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks
#=====================================================================
#Framework: Dynamic Functional Connectivity Network Analysis
#Methodology: Sliding Window based on Derivative Regularity Correlation (SWDRC) and Functional Delay Network (FDN)
#Core Algorithm: Correlation-based on Derivative Regularity (CDR)
#Modality: Resting-state functional Magnetic Resonance Imaging (rs-fMRI)
#Author: Xin Hong, Yongze Lin,and Zhenghao Wu
#Affiliation: Huaqiao University
#Contact: xinhong@hqu.edu.cn
#Version: v1.0.0
#Code Repository: https://github.com/hxpotato/SWDRC
#Copyright © 2025 IEEE
#This code is intended exclusively for academic and research use.
#====================================================================

# -- coding: utf-8 -*-
import os
import numpy as np
from dtaidistance import dtw
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ====== User Configuration Section ======
INPUT_FOLDER  =  # Folder containing original ROI time series txt files
OUTPUT_FOLDER =  # Folder to save DTW distance matrices
N_ROIS        = 90  # Number of ROIs in AAL90 template
MAX_WORKERS   = multiprocessing.cpu_count()  # Number of parallel processes (default: total CPU cores)
# ========================================

def process_file(fname):
    """
    Read a single ROI time series txt file, extract the first N_ROIS columns,
    compute a 90×90 original DTW (Dynamic Time Warping) distance matrix,
    and save the result to the output folder.
    
    Parameters:
        fname (str): Filename of the input ROI time series txt file
    """
    in_path  = os.path.join(INPUT_FOLDER, fname)
    out_path = os.path.join(OUTPUT_FOLDER, fname)

    # 1) Load data and extract the first N_ROIS columns
    data = np.loadtxt(in_path)
    if data.shape[1] < N_ROIS:
        raise ValueError(f"{fname} has fewer than {N_ROIS} columns, cannot extract AAL90 ROIs.")
    data = data[:, :N_ROIS]  # Shape: (Time points, N_ROIS)

    # 2) Initialize output DTW matrix
    dtw_mat = np.zeros((N_ROIS, N_ROIS), dtype=float)

    # 3) Calculate pairwise DTW distances (original implementation)
    for i in range(N_ROIS):
        sig_i = data[:, i]
        for j in range(i, N_ROIS):
            sig_j = data[:, j]
            dist = dtw.distance(sig_i, sig_j)
            dtw_mat[i, j] = dist
            dtw_mat[j, i] = dist  # Symmetric matrix: DTW(i,j) = DTW(j,i)

    # 4) Save the computed DTW matrix
    np.savetxt(out_path, dtw_mat, fmt="%.6f")
    print(f"✔ {fname} ➞ DTW matrix saved to {out_path}")

if __name__ == "__main__":
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Collect all .txt files to process (case-insensitive extension check)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")]
    
    print(f"Number of parallel workers: {MAX_WORKERS}")
    print(f"Total files to process: {len(files)}")
    
    # Parallel execution: each process handles one file
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_file, files)
    
    print("\n✅ All files processed successfully!")
