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
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def load_mats(folder, patient_type, delimiter):
    """
    Load matrices from text files in a folder that match patient type.
    
    Args:
        folder (str): Directory containing the text files
        patient_type (str): Patient type identifier in filename (e.g., 'CN', 'AD')
        delimiter (str): Delimiter used in the text files
    
    Returns:
        numpy.ndarray: Array of loaded matrices
    """
    class_matrices = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt') and patient_type in file_name:
            file_path = os.path.join(folder, file_name)
            z_matrix = np.loadtxt(file_path, delimiter=delimiter)
            class_matrices.append(z_matrix)
    class_matrices = np.array(class_matrices)
    return class_matrices

def extract_significant_features(p_values, threshold):
    """
    Extract indices of features with p-values below the threshold.
    
    Args:
        p_values (numpy.ndarray): Matrix of p-values
        threshold (float): Significance threshold
    
    Returns:
        tuple: Indices of significant features
    """
    significant_features = np.where(p_values < threshold)
    return significant_features

def classify_5z(features, labels):
    """
    Perform 5-fold stratified cross-validation classification.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Class labels
    
    Returns:
        float: Mean accuracy score
    """
    clf = SVC()
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, features, labels, cv=skf)
    print(f"5-fold accuracy: {scores.mean() * 100}%")
    return scores.mean()

def classify_loo(features, labels):
    """
    Perform leave-one-out cross-validation classification.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Class labels
    
    Returns:
        float: Mean accuracy score
    """
    clf = SVC()
    loo = LeaveOneOut()
    scores = cross_val_score(clf, features, labels, cv=loo)
    print(f"Leave-one-out accuracy: {scores.mean() * 100}%")
    return scores.mean()

def main():
    """
    Main function to run the classification pipeline.
    Processes data from multiple folders with different p-value thresholds.
    """
    data_folder = './data/'
    p_val_folder = './pval/'
    threshold = [0.05, 0.01, 0.005, 0.001]
    folders = os.listdir(data_folder)
    
    # Open CSV file for writing results
    csv_file = open('results.csv', 'w')
    csv_file.write("Name,P threshold,Feature count,5-fold,Leave-one-out\n")
    
    for folder in folders:
        name = folder
        print(f"Folder: {name}")
        infolder = os.path.join(data_folder, folder)
        
        # Load control (CN) and Alzheimer's disease (AD) matrices
        CN_mats = load_mats(infolder, 'CN', delimiter=' ')
        AD_mats = load_mats(infolder, 'AD', delimiter=' ')
        
        # Load p-value matrix
        p_val_matrix = np.loadtxt(os.path.join(p_val_folder, f'{name}.txt'), delimiter=' ')
        
        for t in threshold:
            print(f"Threshold: {t}")
            
            # Extract significant features based on p-value threshold
            significant_features = extract_significant_features(p_val_matrix, t)
            feature_count = len(significant_features[0])
            
            # Skip if no significant features found
            if feature_count == 0:
                print("No significant features found, skipping")
                continue
            
            # Extract significant features from CN and AD matrices
            important_data_1 = np.array(CN_mats)[:, significant_features[0], significant_features[1]]
            important_data_2 = np.array(AD_mats)[:, significant_features[0], significant_features[1]]
            important_data = np.concatenate((important_data_1, important_data_2), axis=0)
            
            # Create labels (0 for CN, 1 for AD)
            labels = np.concatenate((np.zeros(CN_mats.shape[0]), np.ones(AD_mats.shape[0])))
            
            # Perform classification and get scores
            score_5z = classify_5z(important_data, labels)
            score_loo = classify_loo(important_data, labels)
            
            # Write results to CSV
            csv_file.write(f"{name},{t},{feature_count},{score_5z},{score_loo}\n")
    
    # Close the CSV file
    csv_file.close()

if __name__ == '__main__':
    main()
