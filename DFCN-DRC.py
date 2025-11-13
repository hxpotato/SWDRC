# =====================================================================
# IEEE Transactions on Medical Imaging (T-MI)
# Unified Cardiac Structure and Pathology Segmentation Framework
# Code Metadata and Implementation Details
# =====================================================================
# Framework: Multi-domain Cardiac MRI Segmentation
# Methodology: Deep Learning-based Unified Segmentation
# Dataset: Multi-center Cardiac Imaging (n=1300+ patients)
# Modality: Cardiac Magnetic Resonance Imaging (MRI)
# Author: Xin Hong, Yongze Lin,and Zhenghao Wu
# Affiliation: HuaQiao University
# Contact: [email/contact information]
# Version: v1.0.0
# Code Repository: [URL to code repository]
# Copyright © 2025 IEEE
# This work is licensed under the MIT License (see LICENSE for details)
# This code is intended exclusively for academic and research use.
# =====================================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Input and output folder paths
in_folder = './ADCN_ROI/'
out_folder = './per/'

def gaussian_kernel(size, sigma):
    """Generate a 1D Gaussian kernel"""
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data, kernel_size, sigma):
    """Apply Gaussian smoothing to 1D data"""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # Normalize kernel
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data

def load_one_file(file_path):
    """Load and transpose matrix from text file"""
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def compute_derivative(series):
    """Calculate numerical derivative of a time series"""
    Dq = np.zeros_like(series)
    for i in range(1, len(series) - 1):
        Dq[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    Dq[0] = Dq[1]  # Handle boundary conditions
    Dq[-1] = Dq[-2]  # Handle boundary conditions
    return Dq

def cal_dis_mat(row1, row2, cols):
    """
    Calculate the distance matrix between two rows
    :param cols: length of the row
    :return: distance matrix with indices from 0 to (cols-1)
    """
    A_deriv = compute_derivative(row1)
    B_deriv = compute_derivative(row2)
    dis = np.zeros((cols, cols))
    for li in range(0, cols):
        for lj in range(0, cols):
            dis[li, lj] = np.sqrt((A_deriv[li] - B_deriv[lj]) ** 2)
    return dis

def dtw_from_distance_matrix(dis_mat):
    """
    Calculate DTW matrix from distance matrix
    :param dis_mat: distance matrix starting from 0
    :return: DTW cumulative cost matrix
    """
    n, m = dis_mat.shape
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dis_mat[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix

def backtrack_dtw(dtw_matrix):
    """
    Find the optimal warping path through DTW matrix
    """
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        # When i == 1, only j can move; when j == 1, only i can move
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        # When both i>1 and j>1, choose the minimum cost direction
        else:
            if dtw_matrix[i-1, j] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
            elif dtw_matrix[i, j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i-1, j-1))
    path.reverse()
    return path

def cal_D(original_matrix, n):
    """
    Calculate Pearson correlation matrix after DTW alignment
    :param original_matrix: original signal matrix
    :param n: number of signals
    :return: correlation matrix
    """
    # Apply Gaussian smoothing
    kernel_size = 9
    sigma = 2
    for i in range(n):
        data = original_matrix[i]
        smoothed_data = gaussian_smooth(data, kernel_size, sigma)
        original_matrix[i] = smoothed_data
    normalized_matrix = original_matrix

    cols = len(original_matrix[0])
    retp = np.zeros((n, n))
    
    # Calculate warped correlation matrix
    for ri in range(0, n):
        for rj in range(0, n):
            if ri == rj:
                retp[ri][rj] = 1
                continue
            if ri > rj:
                continue
            else:
                # Calculate distance matrix between two rows
                dis_mat = cal_dis_mat(normalized_matrix[ri], normalized_matrix[rj], cols)
                # Compute DTW
                dtw = dtw_from_distance_matrix(dis_mat)
                # Find optimal path
                path = backtrack_dtw(dtw)
                # Warp sequences
                aligned_a, aligned_b = warp_sequences(normalized_matrix[ri], normalized_matrix[rj], path)
                newp = pearson_correlation(aligned_a, aligned_b)
                retp[ri][rj] = newp
                retp[rj][ri] = newp
    return retp

def warp_sequences(seq_a, seq_b, path):
    """
    Warp sequences according to DTW path
    """
    aligned_a = []
    aligned_b = []
    for (i, j) in path:
        aligned_a.append(seq_a[i])
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b

def pearson_correlation(seq1, seq2):
    """Calculate Pearson correlation coefficient between two sequences"""
    if np.std(seq1) == 0 or np.std(seq2) == 0:
        print("Warning: Zero standard deviation detected!")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    """Save matrix to text file"""
    np.savetxt(output_path, matrix, delimiter=',')

def process_one_file(filename, in_folder, out_folder):
    """Process a single file"""
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)
    retp = cal_D(matrix, 90)
    save_mat_to_txt(retp, outp_file_path)

def multi_main():
    """Main processing function with progress tracking"""
    for filename in tqdm(os.listdir(in_folder), desc="Processing files"):
        if filename.endswith(".txt"):
            process_one_file(filename, in_folder, out_folder)

if __name__ == '__main__':
    multi_main()


