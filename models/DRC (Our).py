import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dtw_processing.log'),
        logging.StreamHandler()
    ]
)

# Input folder containing AD/CN ROI signal files
INPUT_FOLDER = './ADCN_ROI/'
# Output folder for DTW results with square root transformation
OUTPUT_FOLDER = 'H:/DtwData/dDTW_Ori_sqrt/per/'

# Gaussian smoothing parameters
GAUSSIAN_KERNEL_SIZE = 9  # Must be odd number
GAUSSIAN_SIGMA = 2        # Standard deviation for Gaussian kernel

# Number of sequences (ROIs) per file
NUM_SEQUENCES = 90

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate 1D Gaussian kernel for signal smoothing.
    
    Args:
        size: Kernel window size (should be odd)
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        1D numpy array containing Gaussian kernel values
    """
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data: np.ndarray, kernel_size: int = GAUSSIAN_KERNEL_SIZE, 
                   sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 1D signal with reflection padding to handle boundaries.
    
    Args:
        data: Input 1D signal to be smoothed
        kernel_size: Size of Gaussian kernel (odd integer)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed 1D signal with same length as input
    """
    # Generate and normalize Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # Normalize kernel to sum to 1
    
    # Create mirrored boundary to avoid edge effects
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    
    # Apply convolution and return valid region (same length as input)
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data

def load_one_file(file_path: str) -> np.ndarray:
    """
    Load matrix from text file and return its transpose.
    
    Args:
        file_path: Path to input text file containing matrix data
        
    Returns:
        Transposed numpy array of the loaded matrix
    """
    try:
        matrix = np.loadtxt(file_path)
        transposed_matrix = np.transpose(matrix)
        return transposed_matrix
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise

def compute_derivative(series: np.ndarray) -> np.ndarray:
    """
    Compute first-order derivative of a 1D signal with boundary handling.
    
    Uses central difference for inner points and replicates edge values for boundaries.
    
    Args:
        series: Input 1D signal
        
    Returns:
        Derivative signal with same length as input
    """
    # Initialize derivative array with zeros
    derivative = np.zeros_like(series)
    
    # Compute central difference for inner points
    for i in range(1, len(series) - 1):
        derivative[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    
    # Boundary handling - replicate edge values
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    
    return derivative

def calculate_distance_matrix(row1: np.ndarray, row2: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Calculate distance matrix based on first derivatives of two sequences.
    Uses Euclidean distance (square root of squared differences) between derivatives.
    
    Args:
        row1: First input sequence (1D array)
        row2: Second input sequence (1D array)
        sequence_length: Length of input sequences
        
    Returns:
        Distance matrix (sequence_length x sequence_length)
    """
    # Compute first derivatives of input sequences
    deriv1 = compute_derivative(row1)
    deriv2 = compute_derivative(row2)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((sequence_length, sequence_length))
    
    # Calculate Euclidean distance between all derivative pairs
    for i in range(sequence_length):
        for j in range(sequence_length):
            distance_matrix[i, j] = np.sqrt((deriv1[i] - deriv2[j]) ** 2)
    
    return distance_matrix

def dtw_from_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Dynamic Time Warping (DTW) cost matrix from distance matrix.
    
    The DTW matrix starts at (1,1) with meaningful values, (0,0) = 0 represents
    the starting point with no prior data. All other (0,j) and (i,0) are infinity.
    
    Args:
        distance_matrix: Input distance matrix (n x m)
        
    Returns:
        DTW cost matrix ((n+1) x (m+1))
    """
    n, m = distance_matrix.shape
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n+1, m+1), np.inf)
    # Set starting point cost to 0
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Current cost from distance matrix
            cost = distance_matrix[i-1, j-1]
            # Minimum cost from three possible directions (insert, delete, match)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Insert (up)
                dtw_matrix[i, j-1],    # Delete (left)
                dtw_matrix[i-1, j-1]   # Match (diagonal)
            )
    
    return dtw_matrix

def backtrack_dtw(dtw_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Backtrack through DTW matrix to find optimal alignment path.
    
    Finds the path with minimum total distance between two sequences.
    Path coordinates are already adjusted to match original sequence indices.
    
    Args:
        dtw_matrix: DTW cost matrix from dtw_from_distance_matrix
        
    Returns:
        List of (i,j) tuples representing the optimal alignment path
    """
    # Start from bottom-right corner
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]  # Adjust coordinates to match original sequences
    
    # Backtrack until reaching near start
    while i > 1 or j > 1:
        # Boundary cases - can only move in one direction
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        # Main case - choose direction with minimum cost
        else:
            # Find minimum cost from three possible previous positions
            min_cost = min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1])
            
            if dtw_matrix[i-1, j] == min_cost:
                i -= 1  # Move up
            elif dtw_matrix[i, j-1] == min_cost:
                j -= 1  # Move left
            else:
                i -= 1  # Move diagonally
                j -= 1
        
        path.append((i-1, j-1))
    
    # Reverse path to get forward direction
    path.reverse()
    return path

def warp_sequences(seq_a: np.ndarray, seq_b: np.ndarray, path: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp two sequences according to DTW alignment path.
    
    Important: Do NOT subtract 1 from indices - path coordinates already match
    original sequence indices after backtracking.
    
    Args:
        seq_a: First original sequence (1D array)
        seq_b: Second original sequence (1D array)
        path: DTW alignment path as list of (i,j) tuples
        
    Returns:
        Tuple of aligned sequences (warped_seq_a, warped_seq_b)
    """
    aligned_a = []
    aligned_b = []

    # Align sequences according to path
    for (i, j) in path:
        aligned_a.append(seq_a[i])
        aligned_b.append(seq_b[j])
    
    # Convert to numpy arrays for consistency
    return np.array(aligned_a), np.array(aligned_b)

def pearson_correlation(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two sequences.
    
    Args:
        seq1: First input sequence (1D array)
        seq2: Second input sequence (1D array)
        
    Returns:
        Pearson correlation coefficient (-1 to 1)
        
    Note:
        Returns 0 and logs warning if either sequence has zero standard deviation
    """
    # Check for zero standard deviation (avoid division by zero)
    if np.std(seq1) == 0 or np.std(seq2) == 0:  # Fixed typo (seq1 repeated)
        logging.warning("Zero standard deviation detected in input sequences!")
        return 0.0
    
    try:
        return np.corrcoef(seq1, seq2)[0, 1]
    except Exception as e:
        logging.error(f"Error calculating Pearson correlation: {str(e)}")
        return 0.0

def calculate_dtw_correlation_matrix(original_matrix: np.ndarray, 
                                   num_sequences: int = NUM_SEQUENCES) -> np.ndarray:
    """
    Calculate Pearson correlation matrix using DTW alignment on Gaussian-smoothed sequences.
    
    Args:
        original_matrix: Input matrix with sequences as rows
        num_sequences: Number of sequences (rows) to process
        
    Returns:
        Symmetric Pearson correlation matrix (num_sequences x num_sequences)
    """
    # Apply Gaussian smoothing to all sequences
    smoothed_matrix = original_matrix.copy()
    for i in range(num_sequences):
        smoothed_matrix[i] = gaussian_smooth(original_matrix[i])
    
    sequence_length = len(smoothed_matrix[0])
    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_sequences, num_sequences))
    
    # Set diagonal to 1 (perfect self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Calculate correlation for all unique sequence pairs (i,j) where i < j
    for row_i in tqdm(range(num_sequences), desc="Processing sequences", leave=False):
        for row_j in range(row_i + 1, num_sequences):
            try:
                # Calculate distance matrix based on derivatives
                distance_matrix = calculate_distance_matrix(
                    smoothed_matrix[row_i], 
                    smoothed_matrix[row_j], 
                    sequence_length
                )
                
                # Compute DTW matrix and optimal alignment path
                dtw_matrix = dtw_from_distance_matrix(distance_matrix)
                alignment_path = backtrack_dtw(dtw_matrix)
                
                # Warp sequences according to optimal path
                aligned_seq1, aligned_seq2 = warp_sequences(
                    smoothed_matrix[row_i], 
                    smoothed_matrix[row_j], 
                    alignment_path
                )
                
                # Calculate Pearson correlation for aligned sequences
                correlation = pearson_correlation(aligned_seq1, aligned_seq2)
                
                # Populate symmetric positions in correlation matrix
                correlation_matrix[row_i][row_j] = correlation
                correlation_matrix[row_j][row_i] = correlation
                
            except Exception as e:
                logging.error(f"Error processing pair ({row_i},{row_j}): {str(e)}")
                # Set to 0 if error occurs
                correlation_matrix[row_i][row_j] = 0.0
                correlation_matrix[row_j][row_i] = 0.0
    
    return correlation_matrix

def save_matrix_to_text(matrix: np.ndarray, output_path: str) -> None:
    """
    Save numpy matrix to text file with comma delimiter.
    
    Args:
        matrix: Numpy array to be saved
        output_path: Path to output text file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, matrix, delimiter=',', fmt='%.6f')
        logging.info(f"Successfully saved matrix to {output_path}")
    except Exception as e:
        logging.error(f"Error saving matrix to {output_path}: {str(e)}")
        raise

def process_single_file(file_info: Tuple[str, str, str]) -> None:
    """
    Process a single ROI signal file: load, compute DTW correlation, save results.
    
    Args:
        file_info: Tuple containing (filename, input_folder, output_folder)
    """
    filename, input_folder, output_folder = file_info
    
    try:
        logging.info(f"Starting processing of {filename}")
        
        # Create full file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load and transpose matrix
        signal_matrix = load_one_file(input_path)
        
        # Calculate DTW-based correlation matrix
        correlation_matrix = calculate_dtw_correlation_matrix(signal_matrix)
        
        # Save results
        save_matrix_to_text(correlation_matrix, output_path)
        
        logging.info(f"Completed processing of {filename}")
        
    except Exception as e:
        logging.error(f"Failed to process {filename}: {str(e)}")
        raise

def process_one_file(filename: str, input_folder: str = INPUT_FOLDER, 
                    output_folder: str = OUTPUT_FOLDER) -> None:
    """
    Wrapper function to process a single file with default folder paths.
    
    Args:
        filename: Name of input file
        input_folder: Path to input directory
        output_folder: Path to output directory
    """
    process_single_file((filename, input_folder, output_folder))

def sequential_processing() -> None:
    """Process files one at a time (single-threaded)."""
    # Get list of text files in input folder
    file_list = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    
    # Process each file with progress bar
    for filename in tqdm(file_list, desc="Processing files (sequential)"):
        process_one_file(filename)

def parallel_processing(max_workers: Optional[int] = None) -> None:
    """
    Process files in parallel using multiple CPU cores for faster execution.
    
    Args:
        max_workers: Number of parallel processes (default: number of CPU cores)
    """
    # Set default to number of CPU cores if not specified
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Get list of text files in input folder
    file_list = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    
    # Create list of file info tuples for parallel processing
    file_info_list = [(f, INPUT_FOLDER, OUTPUT_FOLDER) for f in file_list]
    
    logging.info(f"Starting parallel processing with {max_workers} workers")
    
    # Process files in parallel with progress bar
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, info): info[0] 
                         for info in file_info_list}
        
        # Track progress
        for future in tqdm(as_completed(future_to_file), total=len(file_list), 
                          desc="Processing files (parallel)"):
            filename = future_to_file[future]
            try:
                future.result()  # Get result to catch exceptions
            except Exception as e:
                logging.error(f"Parallel processing failed for {filename}: {str(e)}")
    
    logging.info("Parallel processing completed")

def main(use_parallel: bool = True, max_workers: Optional[int] = None) -> None:
    """
    Main function to process all ROI signal files with DTW alignment.
    
    Args:
        use_parallel: If True, use parallel processing (faster for multiple files)
        max_workers: Number of parallel processes (only for parallel mode)
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    logging.info("Starting DTW correlation processing")
    
    try:
        if use_parallel:
            parallel_processing(max_workers)
        else:
            sequential_processing()
        logging.info("All files processed successfully")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    # Run with parallel processing (use all CPU cores)
    # Set use_parallel=False for sequential processing
    main(use_parallel=True, max_workers=None)
