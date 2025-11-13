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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
import argparse

def set_seed(seed=0):
    """Set random seed for reproducibility across all libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA operations
    torch.backends.cudnn.benchmark = False     # Disable benchmarking for reproducibility

def load_data_from_folder(folder):
    """Load connectivity matrices from text files in specified folder
    
    Args:
        folder (str): Path to directory containing .txt matrix files
        
    Returns:
        list: List of numpy arrays containing connectivity matrices
    """
    matrices = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            with open(os.path.join(folder, file)) as f:
                # Read matrix from text file, converting each line to float values
                matrix = np.array([list(map(float, line.split())) for line in f])
                matrices.append(matrix)
    return matrices

def feature_selection(matrix, labels, train_ind, fnum):
    """Perform feature selection using Recursive Feature Elimination (RFE)
    
    Args:
        matrix (numpy.ndarray): Feature matrix (samples × features)
        labels (numpy.ndarray): Class labels for each sample
        train_ind (numpy.ndarray): Indices of training samples
        fnum (int): Number of features to select
        
    Returns:
        selector: Trained RFE selector object
    """
    estimator = RidgeClassifier()  # Use Ridge classifier as base estimator
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    featureX = matrix[train_ind, :]  # Training features
    featureY = labels[train_ind]     # Training labels
    selector = selector.fit(featureX, featureY.ravel())  # Fit selector on training data
    return selector

def flatten_connectivity_matrix(matrix):
    """Flatten symmetric connectivity matrix to upper triangular elements
    
    For an n×n symmetric matrix, extracts n*(n-1)/2 unique features
    by taking elements above the diagonal (excluding self-connections)
    
    Args:
        matrix (numpy.ndarray): Symmetric connectivity matrix
        
    Returns:
        numpy.ndarray: Flattened feature vector of upper triangular elements
    """
    n = matrix.shape[0]  # Matrix dimension (number of nodes)
    features = []
    # Iterate over upper triangular part (i < j)
    for i in range(n):
        for j in range(i+1, n):
            features.append(matrix[i, j])  # Add connection strength value
    return np.array(features)

class Auto_encoder_MLP(nn.Module):
    """Autoencoder + MLP classifier hybrid neural network model
    
    This model combines feature learning capability of autoencoder with
    classification power of MLP, suitable for brain connectivity classification
    with limited samples.
    """
    
    def __init__(self, in_c, auto_1, auto_2, auto_3, MLP_1, MLP_2, MLP_out, dropout_rate=0.5):
        """
        Args:
            in_c (int): Input feature dimension
            auto_1 (int): Autoencoder first hidden layer dimension
            auto_2 (int): Autoencoder second hidden layer dimension
            auto_3 (int): Autoencoder encoding layer dimension (bottleneck)
            MLP_1 (int): Classifier first hidden layer dimension
            MLP_2 (int): Classifier second hidden layer dimension
            MLP_out (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(Auto_encoder_MLP, self).__init__()
        
        # Encoder: Compress input to low-dimensional representation
        self.encoderI'll add comprehensive English comments to the entire code:
