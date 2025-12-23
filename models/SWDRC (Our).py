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
from typing import List, Tuple, Optional

# Input folder containing ROI signal text files
INPUT_FOLDER =
# Output folder to save Pearson correlation matrices
OUTPUT_FOLDER = 

def load_one_file(file_path: str) -> np.ndarray:
    """
    Load a matrix from a text file and return its transpose.
    
    Args:
        file_path: Path to the input text file containing the matrix
        
    Returns:
        Transposed numpy array of the loaded matrix
    """
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def save_mat_to_txt(matrix: np.ndarray, output_path: str) -> None:
    """
    Save a numpy matrix to a text file with space delimiter.
    
    Args:
        matrix: Numpy array to be saved
        output_path: Path to the output text file
    """
    np.savetxt(output_path, matrix, delimiter=' ')

def pearson_correlation(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two sequences.
    
    Args:
        seq1: First input sequence (1D numpy array)
        seq2: Second input sequence (1D numpy array)
        
    Returns:
        Pearson correlation coefficient (float between -1 and 1)
        
    Note:
        Prints warning if either sequence has zero standard deviation
    """
    if np.std(seq1) == 0 or np.std(seq2) == 0:  # Fixed typo (seq1 -> seq2)
        print(f"Warning: Zero standard deviation detected in input sequences!")
    
    # Handle edge case where correlation can't be computed (zero std)
    if np.std(seq1) == 0 or np.std(seq2) == 0:
        return 0.0  # Return 0 instead of NaN
    
    return np.corrcoef(seq1, seq2)[0, 1]

def sliding_window(seq1: np.ndarray, seq2: np.ndarray, 
                   window_size: int = 10, stride: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate aligned sequence pairs using sliding window approach.
    
    Args:
        seq1: First input sequence (1D numpy array)
        seq2: Second input sequence (1D numpy array)
        window_size: Size of the sliding window (default: 10)
        stride: Step size for window movement (default: 1)
        
    Returns:
        List of tuples containing aligned window pairs from seq1 and seq2
        
    Raises:
        ValueError: If input sequences have different lengths
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Input sequences must have the same length! "
                         f"Got {len(seq1)} and {len(seq2)}")

    all_aligned_pairs = []

    # Iterate through sequence with sliding window
    for start in range(0, len(seq1) - window_size + 1, stride):
        end = start + window_size
        
        # Extract window segments from both sequences
        window_seq1 = seq1[start:end]
        window_seq2 = seq2[start:end]
        
        all_aligned_pairs.append((window_seq1, window_seq2))

    return all_aligned_pairs

def calculate_window_correlation(original_matrix: np.ndarray, 
                                num_sequences: int,
                                window_size: int = 10, 
                                stride: int = 1) -> np.ndarray:
    """
    Calculate average Pearson correlation matrix using sliding window approach.
    
    Computes average Pearson correlation between all pairs of sequences 
    across sliding windows, resulting in a symmetric correlation matrix.
    
    Args:
        original_matrix: Input matrix where each row is a sequence
        num_sequences: Number of sequences (rows) to process
        window_size: Size of the sliding window (default: 10)
        stride: Step size for window movement (default: 1)
        
    Returns:
        Symmetric correlation matrix (num_sequences x num_sequences)
    """
    # Initialize correlation matrix with zeros
    correlation_matrix = np.zeros((num_sequences, num_sequences))
    
    # Calculate correlation for each unique pair (i,j) where i < j
    for row_i in range(num_sequences):
        for row_j in range(row_i + 1, num_sequences):
            # Get full-length sequences from matrix rows
            seq1 = original_matrix[row_i]
            seq2 = original_matrix[row_j]
            
            # Generate sliding window pairs
            aligned_pairs = sliding_window(seq1, seq2, window_size, stride)
            
            # Calculate Pearson correlation for each window
            window_correlations = [pearson_correlation(pair[0], pair[1]) 
                                  for pair in aligned_pairs]
            
            # Calculate average correlation across all windows
            avg_correlation = np.mean(window_correlations)
            
            # Populate symmetric positions in correlation matrix
            correlation_matrix[row_i][row_j] = avg_correlation
            correlation_matrix[row_j][row_i] = avg_correlation
    
    # Set diagonal elements to 1 (perfect self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    return correlation_matrix

def main() -> None:
    """
    Main processing function:
    1. Iterates through all text files in input folder
    2. Loads each ROI signal matrix
    3. Computes sliding window Pearson correlation matrix
    4. Saves result to output folder with standardized naming
    
    Progress bar shows processing status
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Process each text file in input folder with progress bar
    for filename in tqdm(os.listdir(INPUT_FOLDER), 
                         desc="Processing ROI files",
                         position=0,
                         leave=True):
        
        # Only process .txt files
        if filename.endswith(".txt"):
            # Construct full file paths
            input_file_path = os.path.join(INPUT_FOLDER, filename)
            output_file_path = os.path.join(OUTPUT_FOLDER, f'Per{filename}')
            
            try:
                # Load and process matrix
                signal_matrix = load_one_file(input_file_path)
                
                # Calculate correlation matrix (90 sequences, window size 10, stride 1)
                correlation_result = calculate_window_correlation(
                    signal_matrix, 
                    num_sequences=90,
                    window_size=10,
                    stride=1
                )
                
                # Save result to output file
                save_mat_to_txt(correlation_result, output_file_path)
                
            except Exception as e:
                # Handle errors for individual files without stopping entire process
                print(f"\nError processing {filename}: {str(e)}")
                continue

if __name__ == '__main__':
    # Execute main function when script is run directly
    main()
