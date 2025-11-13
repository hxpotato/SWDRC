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

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import numpy as np
import os

# Input and output folder paths
in_folder = './ADCN_ROI/'
out_folder = './out_folder/'

# Window sizes and stride parameters
window_size = [i for i in range(10, 55, 5)]
# Previous squared calculations completed, now calculating square root results
stride = [i for i in range(1, 11)]
# stride = [i for i in range(1, 5)] Completed
# stride = [i for i in range(5, 9)] Completed
# stride = [i for i in range(9, 11)] Completed

def gaussian_kernel(size, sigma):
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        size: Size of the kernel
        sigma: Standard deviation of the Gaussian distribution
    
    Returns:
        Gaussian kernel array
    """
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data, kernel_size, sigma):
    """
    Apply Gaussian smoothing to data.
    
    Args:
        data: Input data to be smoothed
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian distribution
    
    Returns:
        Smoothed data
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # Normalize kernel
    # Create mirrored boundaries
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data

def load_one_file(file_path):
    """
    Load and transpose a matrix from a text file.
    
    Args:
        file_path: Path to the input file
    
    Returns:
        Transposed matrix
    """
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def compute_derivative(series):
    """
    Calculate the first derivative of a series.
    
    Args:
        series: Input time series data
    
    Returns:
        Derivative of the series
    """
    Dq = np.zeros_like(series)
    for i in range(1, len(series) - 1):
        Dq[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    Dq[0] = Dq[1]  # Boundary handling
    Dq[-1] = Dq[-2]  # Boundary handling
    return Dq

def cal_dis_mat(row1, row2, cols):
    """
    Calculate distance matrix between two rows.
    
    Args:
        row1: First row of data
        row2: Second row of data
        cols: Length of the rows
    
    Returns:
        Distance matrix with indices 0-(cols-1)
    """
    # Calculate derivatives
    A_deriv = compute_derivative(row1)
    B_deriv = compute_derivative(row2)
    dis = np.zeros((cols, cols))
    for li in range(0, cols):
        for lj in range(0, cols):
            # Calculate distance matrix between derivatives
            dis[li, lj] = np.sqrt((A_deriv[li] - B_deriv[lj]) ** 2)
    return dis

def dtw_from_distance_matrix(dis_mat):
    """
    Calculate DTW matrix from distance matrix.
    The matrix starts from position (1,1) with actual meaning, (0,0) is set to 0,
    representing the cumulative cost when both sequences start from their initial positions.
    
    Args:
        dis_mat: Distance matrix starting from 0
    
    Returns:
        DTW matrix
    """
    n, m = dis_mat.shape
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Note: positions where i=0 or j=0 are set to infinity, so when (u=1, v>1) or (u>1, v=1)
    # the algorithm won't transfer from u=0 or v=0 positions
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dis_mat[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix

def backtrack_dtw(dtw_matrix):
    """
    Find the path that minimizes the total distance between two sequences.
    Note: The path indices are already -1, no need to subtract again in warp_sequences.
    
    Args:
        dtw_matrix: DTW matrix
    
    Returns:
        Optimal path through the DTW matrix
    """
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        # When i == 1, only move j // When j == 1, only move i
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        # When i>1, j>1, there are three movement directions, choose the smallest one
        else:
            # Move up if it's the smallest direction
            if dtw_matrix[i-1, j] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
            # Move left if it's the smallest direction
            elif dtw_matrix[i, j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                j -= 1
            # Move diagonally if it's the smallest direction
            else:
                i -= 1
                j -= 1
        path.append((i-1, j-1))
    path.reverse()
    return path

def sliding_window_DTW(seq1, seq2, window_size=10, stride=1):
    """
    Calculate DTW between two sequences within sliding windows and return aligned sequences for each window.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        window_size: Size of the sliding window
        stride: Number of positions the window moves each iteration
    
    Returns:
        Tuple containing list of aligned sequence pairs and path calculation matrix
    """
    all_aligned_pairs = []
    # Uncomment if storing sequence alignment paths
    # paths = []
    # Initialize path calculation matrix
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
    """
    Calculate Pearson correlation coefficients between signals in the original signal matrix before and after warping.
    
    Args:
        ormat: Original signal matrix
        n: Number of signals
        window_size: Size of the sliding window
        stride: Stride for window movement
        path_cal_save_folder: Folder to save path calculation results
        patient_id: Patient identifier
    
    Returns:
        Tuple containing correlation matrix and path calculation matrix
    """
    # ## Note: Different normalization methods can be tried, such as z-score normalization
    # # Calculate row totals
    # row_means = ormat.sum(axis=1, keepdims=True)
    # # Normalize each element by its row total
    # normalized_matrix = ormat / row_means

    # Apply Gaussian smoothing
    kernel_size = 9
    # kernel_size = 5
    # kernel_size needs to be odd, sigma is standard deviation
    sigma = 2
    for i in range(n):
        data = ormat[i]
        smoothed_data = gaussian_smooth(data, kernel_size, sigma)
        ormat[i] = smoothed_data
    # Perform normalization on the entire sequence

    # cols = len(ormat[0])
    retp = np.zeros((n, n))
    # Path calculation matrix records matching counts in warping paths
    ALL_ROI_path_cal = np.zeros((n, n))
    # Note: Need to record internally as it's for calculations between different ROIs
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
                # Pass the full-length sequences
                aligned_pairs, t_path_cal = sliding_window_DTW(ormat[ri], ormat[rj], window_size, stride)
                # Calculate Pearson correlation coefficients from all aligned sequences
                local_pearsons = [pearson_correlation(pair[0], pair[1]) for pair in aligned_pairs]
                # Calculate and save average Pearson correlation coefficient
                retp[ri][rj] = np.mean(local_pearsons)
                retp[rj][ri] = retp[ri][rj]

                # Calculate matching counts in warping paths
                path_cal_sum = np.sum(t_path_cal)
                # Calculate corresponding row and column sums
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
                # Don't store, calculate path directly

    return retp, ALL_ROI_path_cal

def warp_sequences(seq_a, seq_b, path):
    """
    Warp sequences based on DTW path.
    
    Args:
        seq_a: First original sequence
        seq_b: Second original sequence
        path: Warping path between the two sequences
        path: (1,1),(1,2),(2,3)...(n,n)
    
    Returns:
        Tuple of warped sequences
    """
    aligned_a = []
    aligned_b = []

    for (i, j) in path:
        aligned_a.append(seq_a[i])  # Subtract 1 because path is based on dtw_matrix which is larger than seq_a and seq_b by 1
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b

def pearson_correlation(seq1, seq2):
    """Calculate Pearson correlation coefficient between two sequences."""
    if np.std(seq1) == 0 or np.std(seq2) == 0:
        print("Sequence with zero standard deviation found!")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    """Save matrix to text file."""
    np.savetxt(output_path, matrix, delimiter=' ')

def process_one_file(filename, window_size, stride, in_folder, out_folder):
    """
    Process a single file: load, calculate DTW, and save results.
    
    Args:
        filename: Name of the file to process
        window_size: Window size for DTW calculation
        stride: Stride for window movement
        in_folder: Input folder path
        out_folder: Output folder path
    """
    # Define input and output file paths, load matrix (already transposed)
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)

    path_cal_save_folder = out_folder + 'path_cal/'
    if not os.path.exists(path_cal_save_folder):
        os.makedirs(path_cal_save_folder)

    patient_id = filename.replace('ROISignal_', '')
    patient_id = patient_id.replace('_CN.txt', '')
    patient_id = patient_id.replace('_AD.txt', '')

    # Calculate Pearson correlation coefficient matrix after warping and save
    retp, path_cal = cal_window_D(matrix, 90, window_size, stride, path_cal_save_folder, patient_id)
    save_mat_to_txt(retp, outp_file_path)
    path_output_path = os.path.join(out_folder, filename + '_path_cal.txt')
    save_mat_to_txt(path_cal, path_output_path)

def main():
    """Main function to process all files with different window sizes and strides."""
    # Iterate through all files in the folder
    for filename in os.listdir(in_folder):
        if filename.endswith(".txt"):
            for ws in window_size:
                for st in stride:
                    # Iterate through window sizes and strides
                    now_WDO_folder = os.path.join(out_folder, f"{ws}_{st}")
                    if not os.path.exists(now_WDO_folder):
                        os.makedirs(now_WDO_folder)
                    # Directly call processing function
                    process_one_file(filename, ws, st, in_folder, now_WDO_folder)
                    print(f"Processed file {filename} completed!")

if __name__ == '__main__':
    main()

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import numpy as np
import os

# Input and output folder paths
in_folder = './ADCN_ROI/'
out_folder = 'H:/DtwData/dDTW_Ori_sqrt_new/WDO_per1111/'

# Window sizes and stride parameters
window_size = [i for i in range(10, 55, 5)]
# Previous squared calculations completed, now calculating square root results
stride = [i for i in range(1, 11)]
# stride = [i for i in range(1, 5)] Completed
# stride = [i for i in range(5, 9)] Completed
# stride = [i for i in range(9, 11)] Completed

def gaussian_kernel(size, sigma):
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        size: Size of the kernel
        sigma: Standard deviation of the Gaussian distribution
    
    Returns:
        Gaussian kernel array
    """
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data, kernel_size, sigma):
    """
    Apply Gaussian smoothing to data.
    
    Args:
        data: Input data to be smoothed
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian distribution
    
    Returns:
        Smoothed data
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # Normalize kernel
    # Create mirrored boundaries
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data

def load_one_file(file_path):
    """
    Load and transpose a matrix from a text file.
    
    Args:
        file_path: Path to the input file
    
    Returns:
        Transposed matrix
    """
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def compute_derivative(series):
    """
    Calculate the first derivative of a series.
    
    Args:
        series: Input time series data
    
    Returns:
        Derivative of the series
    """
    Dq = np.zeros_like(series)
    for i in range(1, len(series) - 1):
        Dq[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    Dq[0] = Dq[1]  # Boundary handling
    Dq[-1] = Dq[-2]  # Boundary handling
    return Dq

def cal_dis_mat(row1, row2, cols):
    """
    Calculate distance matrix between two rows.
    
    Args:
        row1: First row of data
        row2: Second row of data
        cols: Length of the rows
    
    Returns:
        Distance matrix with indices 0-(cols-1)
    """
    # Calculate derivatives
    A_deriv = compute_derivative(row1)
    B_deriv = compute_derivative(row2)
    dis = np.zeros((cols, cols))
    for li in range(0, cols):
        for lj in range(0, cols):
            # Calculate distance matrix between derivatives
            dis[li, lj] = np.sqrt((A_deriv[li] - B_deriv[lj]) ** 2)
    return dis

def dtw_from_distance_matrix(dis_mat):
    """
    Calculate DTW matrix from distance matrix.
    The matrix starts from position (1,1) with actual meaning, (0,0) is set to 0,
    representing the cumulative cost when both sequences start from their initial positions.
    
    Args:
        dis_mat: Distance matrix starting from 0
    
    Returns:
        DTW matrix
    """
    n, m = dis_mat.shape
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Note: positions where i=0 or j=0 are set to infinity, so when (u=1, v>1) or (u>1, v=1)
    # the algorithm won't transfer from u=0 or v=0 positions
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dis_mat[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix

def backtrack_dtw(dtw_matrix):
    """
    Find the path that minimizes the total distance between two sequences.
    Note: The path indices are already -1, no need to subtract again in warp_sequences.
    
    Args:
        dtw_matrix: DTW matrix
    
    Returns:
        Optimal path through the DTW matrix
    """
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        # When i == 1, only move j // When j == 1, only move i
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        # When i>1, j>1, there are three movement directions, choose the smallest one
        else:
            # Move up if it's the smallest direction
            if dtw_matrix[i-1, j] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
            # Move left if it's the smallest direction
            elif dtw_matrix[i, j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                j -= 1
            # Move diagonally if it's the smallest direction
            else:
                i -= 1
                j -= 1
        path.append((i-1, j-1))
    path.reverse()
    return path

def sliding_window_DTW(seq1, seq2, window_size=10, stride=1):
    """
    Calculate DTW between two sequences within sliding windows and return aligned sequences for each window.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        window_size: Size of the sliding window
        stride: Number of positions the window moves each iteration
    
    Returns:
        Tuple containing list of aligned sequence pairs and path calculation matrix
    """
    all_aligned_pairs = []
    # Uncomment if storing sequence alignment paths
    # paths = []
    # Initialize path calculation matrix
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
    """
    Calculate Pearson correlation coefficients between signals in the original signal matrix before and after warping.
    
    Args:
        ormat: Original signal matrix
        n: Number of signals
        window_size: Size of the sliding window
        stride: Stride for window movement
        path_cal_save_folder: Folder to save path calculation results
        patient_id: Patient identifier
    
    Returns:
        Tuple containing correlation matrix and path calculation matrix
    """
    # ## Note: Different normalization methods can be tried, such as z-score normalization
    # # Calculate row totals
    # row_means = ormat.sum(axis=1, keepdims=True)
    # # Normalize each element by its row total
    # normalized_matrix = ormat / row_means

    # Apply Gaussian smoothing
    kernel_size = 9
    # kernel_size = 5
    # kernel_size needs to be odd, sigma is standard deviation
    sigma = 2
    for i in range(n):
        data = ormat[i]
        smoothed_data = gaussian_smooth(data, kernel_size, sigma)
        ormat[i] = smoothed_data
    # Perform normalization on the entire sequence

    # cols = len(ormat[0])
    retp = np.zeros((n, n))
    # Path calculation matrix records matching counts in warping paths
    ALL_ROI_path_cal = np.zeros((n, n))
    # Note: Need to record internally as it's for calculations between different ROIs
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
                # Pass the full-length sequences
                aligned_pairs, t_path_cal = sliding_window_DTW(ormat[ri], ormat[rj], window_size, stride)
                # Calculate Pearson correlation coefficients from all aligned sequences
                local_pearsons = [pearson_correlation(pair[0], pair[1]) for pair in aligned_pairs]
                # Calculate and save average Pearson correlation coefficient
                retp[ri][rj] = np.mean(local_pearsons)
                retp[rj][ri] = retp[ri][rj]

                # Calculate matching counts in warping paths
                path_cal_sum = np.sum(t_path_cal)
                # Calculate corresponding row and column sums
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
                # Don't store, calculate path directly

    return retp, ALL_ROI_path_cal

def warp_sequences(seq_a, seq_b, path):
    """
    Warp sequences based on DTW path.
    
    Args:
        seq_a: First original sequence
        seq_b: Second original sequence
        path: Warping path between the two sequences
        path: (1,1),(1,2),(2,3)...(n,n)
    
    Returns:
        Tuple of warped sequences
    """
    aligned_a = []
    aligned_b = []

    for (i, j) in path:
        aligned_a.append(seq_a[i])  # Subtract 1 because path is based on dtw_matrix which is larger than seq_a and seq_b by 1
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b

def pearson_correlation(seq1, seq2):
    """Calculate Pearson correlation coefficient between two sequences."""
    if np.std(seq1) == 0 or np.std(seq2) == 0:
        print("Sequence with zero standard deviation found!")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    """Save matrix to text file."""
    np.savetxt(output_path, matrix, delimiter=' ')

def process_one_file(filename, window_size, stride, in_folder, out_folder):
    """
    Process a single file: load, calculate DTW, and save results.
    
    Args:
        filename: Name of the file to process
        window_size: Window size for DTW calculation
        stride: Stride for window movement
        in_folder: Input folder path
        out_folder: Output folder path
    """
    # Define input and output file paths, load matrix (already transposed)
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)

    path_cal_save_folder = out_folder + 'path_cal/'
    if not os.path.exists(path_cal_save_folder):
        os.makedirs(path_cal_save_folder)

    patient_id = filename.replace('ROISignal_', '')
    patient_id = patient_id.replace('_CN.txt', '')
    patient_id = patient_id.replace('_AD.txt', '')

    # Calculate Pearson correlation coefficient matrix after warping and save
    retp, path_cal = cal_window_D(matrix, 90, window_size, stride, path_cal_save_folder, patient_id)
    save_mat_to_txt(retp, outp_file_path)
    path_output_path = os.path.join(out_folder, filename + '_path_cal.txt')
    save_mat_to_txt(path_cal, path_output_path)

def main():
    """Main function to process all files with different window sizes and strides."""
    # Iterate through all files in the folder
    for filename in os.listdir(in_folder):
        if filename.endswith(".txt"):
            for ws in window_size:
                for st in stride:
                    # Iterate through window sizes and strides
                    now_WDO_folder = os.path.join(out_folder, f"{ws}_{st}")
                    if not os.path.exists(now_WDO_folder):
                        os.makedirs(now_WDO_folder)
                    # Directly call processing function
                    process_one_file(filename, ws, st, in_folder, now_WDO_folder)
                    print(f"Processed file {filename} completed!")

if __name__ == '__main__':
    main()


