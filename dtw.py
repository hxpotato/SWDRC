#=====================================================================
#IEEE Transactions on Medical Imaging (T-MI)
#Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks
#=====================================================================
#Framework: Dynamic Functional Connectivity Network Analysis
#Methodology: Sliding Window based on Derivative Regularity Correlation (SWDRC) and Functional Delay Network (FDN)
#Core Algorithm: Correlation-based on Derivative Regularity (CDR)
#Dataset: ADNI (n=417 subjects), ABIDE (NYU & UM sites)
#Modality: Resting-state functional Magnetic Resonance Imaging (rs-fMRI)
#Author: Xin Hong, Yongze Lin,and Zhenghao Wu
#Affiliation: Huaqiao University
#Contact: xinhong@hqu.edu.cn
#Version: v1.0.0
#Code Repository: https://github.com/hxpotato/SWDRC
#Copyright © 2025 IEEE
#This work is licensed under the MIT License (see LICENSE for details)
#This code is intended exclusively for academic and research use.
#====================================================================

# -*- coding: utf-8 -*-
import os
import numpy as np
from dtaidistance import dtw
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ====== User Configuration Area ======
input_folder  = "G:/WPSData/Code/TMI_lastexp/roi_data/data_EMCI_CN"   # Folder containing original ROI time series txt files
output_folder = "G:/WPSData/Code/TMI_lastexp/out_put_all/ori_dtw/EMCI_CN"  # Folder to save the output DTW matrices
OUTPUT_FOLDER = output_folder
INPUT_FOLDER = input_folder
N_ROIS        = 90                             # Number of ROIs according to AAL90 atlas
MAX_WORKERS   = multiprocessing.cpu_count()    # Number of parallel processes, defaults to CPU core count
# ======================================

def process_file(fname):
    """Read a single txt file, extract the first N_ROIS columns, compute the 90×90 raw DTW distance matrix, and save the output."""
    in_path  = os.path.join(INPUT_FOLDER, fname)
    out_path = os.path.join(OUTPUT_FOLDER, fname)

    # 1) Load data and extract the first N_ROIS columns
    data = np.loadtxt(in_path)
    if data.shape[1] < N_ROIS:
        raise ValueError(f"{fname} has less than {N_ROIS} columns, cannot extract AAL90 data.")
    data = data[:, :N_ROIS]  # Shape (T, N_ROIS)

    # 2) Initialize output matrix
    dtw_mat = np.zeros((N_ROIS, N_ROIS), dtype=float)

    # 3) Pairwise DTW distance calculation (raw implementation)
    for i in range(N_ROIS):
        sig_i = data[:, i]
        for j in range(i, N_ROIS):
            sig_j = data[:, j]
            dist = dtw.distance(sig_i, sig_j)
            dtw_mat[i, j] = dist
            dtw_mat[j, i] = dist

    # 4) Save result
    np.savetxt(out_path, dtw_mat, fmt="%.6f")
    print(f"✔ {fname} ➞➞ DTW matrix saved to {out_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    # Collect all .txt file names to be processed
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")]
    print(MAX_WORKERS)
    # Parallel distribution: each process handles one file
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_file, files)


