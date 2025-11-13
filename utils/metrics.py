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
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob_pos: np.ndarray) -> dict:
    """
    Calculate key evaluation metrics for binary classification tasks:
    Accuracy (ACC), AUC, Sensitivity (Recall), Specificity.
    
    Parameters:
        y_true (np.ndarray): 1D array of true labels (0 = negative class, 1 = positive class)
        y_pred (np.ndarray): 1D array of predicted class labels (0 or 1)
        y_pred_prob_pos (np.ndarray): 1D array of predicted probabilities for the positive class (range: [0,1])
    
    Returns:
        dict: Dictionary containing all calculated metrics with keys:
              'accuracy', 'auc', 'sensitivity', 'specificity'
    """
    # Calculate confusion matrix: TN, FP, FN, TP
    # Confusion matrix structure (sklearn default):
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Flatten confusion matrix to extract components
    
    # 1. Accuracy (ACC): (TP + TN) / Total samples
    total_samples = len(y_true)
    accuracy = (tp + tn) / total_samples if total_samples != 0 else 0.0
    
    # 2. Sensitivity (Recall): TP / (TP + FN) - Ability to correctly identify positive cases
    # Handle edge case: no positive samples in true labels (TP + FN = 0)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    
    # 3. Specificity: TN / (TN + FP) - Ability to correctly identify negative cases
    # Handle edge case: no negative samples in true labels (TN + FP = 0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    
    # 4. AUC (Area Under ROC Curve): Measures model's ability to distinguish between classes
    # Handle edge case: all true labels are the same (only one class present)
    if len(np.unique(y_true)) < 2:
        auc_score = 0.0
    else:
        auc_score = roc_auc_score(y_true, y_pred_prob_pos)
    
    # Return metrics as a dictionary with rounded values (4 decimal places for readability)
    return {
        'accuracy': round(accuracy, 4),
        'auc': round(auc_score, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4)
    }


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate sample data (simulate binary classification results)
    np.random.seed(42)  # For reproducibility
    
    # True labels (50% positive, 50% negative)
    y_true = np.random.randint(0, 2, size=1000)
    
    # Predicted class labels (slightly better than random)
    y_pred = np.where(np.random.rand(1000) > 0.3, y_true, 1 - y_true)
    
    # Predicted probabilities for positive class (simulate model confidence)
    y_pred_prob_pos = np.where(y_true == 1, 
                               np.random.uniform(0.4, 1.0, size=1000),
                               np.random.uniform(0.0, 0.6, size=1000))
    
    # Calculate metrics
    metrics = calculate_evaluation_metrics(y_true, y_pred, y_pred_prob_pos)
    
    # Print results
    print("Evaluation Metrics for Binary Classification:")
    print(f"Accuracy (ACC): {metrics['accuracy']}")
    print(f"AUC: {metrics['auc']}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']}")
    print(f"Specificity: {metrics['specificity']}")
