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

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
# Importing numpy and os again is redundant; consider removing the duplicates.

# Define input and output folder paths
in_folder = './ADCN_ROI/'
out_folder = 'H:/DtwData/dDTW_Ori_sqrt_new/WDO_per1111/'

# Define window sizes to be used in sliding window DTW
window_size = [i for i in range(10, 55, 5)]
# The original square has been completed; now compute results under square root
stride = [i for i in range(1, 11)]
# stride = [i for i in range(1, 5)] # Completed
# stride = [i for i in range(5, 9)] # Completed
# stride = [i for i in range(9, 11)] # Completed

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel of given size and standard deviation (sigma)."""
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data, kernel_size, sigma):
    """Apply Gaussian smoothing to data using the specified kernel size and sigma."""
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # Normalize the kernel
    # Create mirrored boundaries
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data

def load_one_file(file_path):
    """Load matrix from file and transpose it."""
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def compute_derivative(series):
    """Compute the first derivative of the series with boundary handling."""
    Dq = np.zeros_like(series)
    for i in range(1, len(series) - 1):
        Dq[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    Dq[0] = Dq[1]  # Boundary handling
    Dq[-1] = Dq[-2]  # Boundary handling
    return Dq

def cal_dis_mat(row1, row2, cols):
    """Calculate distance matrix between two rows based on their derivatives."""
    A_deriv = compute_derivative(row1)
    B_deriv = compute_derivative(row2)
    dis = np.zeros((cols, cols))
    for li in range(0, cols):
        for lj in range(0, cols):
            # Calculate distance matrix between derivatives
            dis[li, lj] = np.sqrt((A_deriv[li] - B_deriv[lj]) ** 2)
    return dis

def dtw_from_distance_matrix(dis_mat):
    """Compute DTW matrix from the distance matrix."""
    n, m = dis_mat.shape
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    # Note that where i=0 or j=0 are set to infinity, so when (u=1, v>1) or (u>1, v=1), they won't transfer from u=0 or v=0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dis_mat[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix

def backtrack_dtw(dtw_matrix):
    """Find the path minimizing total distance between two sequences."""
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        # When i == 1, only move j // When j == 1, only move i
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        # When both i > 1 and j > 1, choose the minimum value among three possible moves
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

def sliding_window_DTW(seq1, seq2, window_size=10, stride=1):
    """
    Compute DTW within a sliding window for two sequences and return aligned sequences for each window.

    Parameters:
    :param seq1: First sequence.
    :param seq2: Second sequence.
    :param window_size: Size of the sliding window.
    :param stride: Number of positions to slide the window at each step.
    :param path_cal: Record matching count of warping paths.

    Returns:
    :return: A list of tuples containing aligned sequences for each window.
    """
    all_aligned_pairs = []
    # If storing sequence alignment paths, uncomment below
    # paths = []
    # Initialize warping path calculation matrix
    path_cal = np.zeros((len(seq1)+5, len(seq2)+5))

    for start in range(0, len(seq1) - window_size + 1, stride):
        end = start + window_size

        normalized_seq1 = seq1[start:end]
        normalized_seq2 = seq2[start:end]

        dis_mat = cal_dis_mat(normalized_seq1, normalized_seq2, end - start)
        dtw_matrix = dtw_from_distance_matrix(dis_mat)
        path = backtrack_dtw(dtw_matrix)

        for (i, j) in path:
            path_cal[i, j] += 1
        # paths.append(path)
        aligned_a, aligned_b = warp_sequences(normalized_seq1, normalized_seq2, path)
        all_aligned_pairs.append((aligned_a, aligned_b))

    # return all_aligned_pairs, paths
    return all_aligned_pairs, path_cal

def cal_window_D(ormat, n, window_size, stride, path_cal_save_folder, patient_id):
    '''
    Calculate Pearson correlation coefficients before and after warping between signals in the raw signal matrix.
    :param ormat: Original signal matrix
    :param n: Number of signals
    '''
    # Gaussian smoothing
    kernel_size = 9
    # kernel_size = 5
    # kernel_size needs to be odd, sigma is standard deviation
    sigma = 2
    for i in range(n):
        data = ormat[i]
        smoothed_data = gaussian_smooth(data, kernel_size, sigma)
        ormat[i] = smoothed_data
    # Perform normalization on the entire sequence first

    retp = np.zeros((n, n))
    # path_cal matrix records matching counts of warping paths
    ALL_ROI_path_cal = np.zeros((n, n))
    # !!! Note, need to record inside because calculations are done between different ROIs
    # Calculate warping path matrix D
    # For row i and row j
    for ri in range(0, n):
        for rj in range(0, n):
            if ri == rj:
                retp[ri][rj] = 1
                continue
            if ri > rj:
                continue
            else:
                # Pass full length sequences
                aligned_pairs, t_path_cal = sliding_window_DTW(ormat[ri], ormat[rj], window_size, stride)
                # Calculate Pearson correlation coefficients based on all aligned sequences
                local_pearsons = [pearson_correlation(pair[0], pair[1]) for pair in aligned_pairs]
                # Compute and save average Pearson correlation coefficient
                retp[ri][rj] = np.mean(local_pearsons)
                retp[rj][ri] = retp[ri][rj]

                # Calculate matching counts of warping paths
                path_cal_sum = np.sum(t_path_cal)
                # Sum corresponding rows and columns
                ri_cal = 0.0
                rj_cal = 0.0
                sqe_len = len(ormat[ri])
                # Note: path is (0,0),(0,1),(1,2)...(n,n)
                for tmp_i in range(sqe_len+3):
                    for tmp_j in range(sqe_len+3):
                        ri_cal = ri_cal + (tmp_i * t_path_cal[tmp_i][tmp_j])
                        rj_cal = rj_cal + (tmp_j * t_path_cal[tmp_i][tmp_j])

                # Calculate mean difference
                path_val = (ri_cal - rj_cal) / (path_cal_sum)
                ALL_ROI_path_cal[ri][rj] = path_val
                ALL_ROI_path_cal[rj][ri] = -path_val

                # path_output_path = os.path.join(path_cal_save_folder, patient_id+'_'+str(ri)+'_'+str(rj)+'.txt')
                # save_mat_to_txt(path_cal, path_output_path)
                # Do not store, calculate path directly

    return retp, ALL_ROI_path_cal

def warp_sequences(seq_a, seq_b, path):
    '''
    seq_a, seq_b: Two original sequences
    path: Warping path between the two sequences
    path：(1,1),(1,2),(2,3)...(n,n)
    Restore warped sequences
    '''
    aligned_a = []
    aligned_b = []

    for (i, j) in path:
        aligned_a.append(seq_a[i])  # Subtract 1 because path is based on dtw_matrix which is larger by 1 than seq_a and seq_b
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b

def pearson_correlation(seq1, seq2):
    """Calculate Pearson correlation coefficient between two sequences."""
    if(np.std(seq1)==0 or np.std(seq1)==0):
        print("0000！")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    """Save matrix to text file."""
    np.savetxt(output_path, matrix, delimiter=' ')

def process_one_file(filename, window_size, stride, in_folder, out_folder):
    # Define input and output file paths, read matrix (already transposed)
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)

    path_cal_save_folder = out_folder + 'path_cal/'
    if not os.path.exists(path_cal_save_folder):
        os.makedirs(path_cal_save_folder)

    patient_id = filename.replace('ROISignal_', '')
    patient_id = patient_id.replace('_CN.txt', '')
    patient_id = patient_id.replace('_AD.txt', '')

    # Calculate and save warped Pearson correlation coefficient matrix
    retp, path_cal = cal_window_D(matrix, 90, window_size, stride, path_cal_save_folder, patient_id)
    save_mat_to_txt(retp, outp_file_path)
    path_output_path = os.path.join(out_folder, filename + '_path_cal.txt')
    save_mat_to_txt(path_cal, path_output_path)

def main():
    # Iterate over all files in the directory
    for filename in os.listdir(in_folder):
        if filename.endswith(".txt"):
            for ws in window_size:
                for st in stride:
                    # Iterate over window sizes and strides
                    now_WDO_folder = os.path.join(out_folder, f"{ws}_{st}")
                    if not os.path.exists(now_WDO_folder):
                        os.makedirs(now_WDO_folder)
                    # Directly call processing function
                    process_one_file(filename, ws, st, in_folder, now_WDO_folder)
                    print(f"Processing file {filename} complete!")

if __name__ == '__main__':
    main()
