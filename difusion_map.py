#=====================================================================
#IEEE Transactions on Medical Imaging (T-MI)
#Unified Cardiac Structure and Pathology Segmentation Framework
#Code Metadata and Implementation Details
#=====================================================================
#Framework: Multi-domain Cardiac MRI Segmentation
#Methodology: Deep Learning-based Unified Segmentation
#Dataset: Multi-center Cardiac Imaging (n=1300+ patients)
#Modality: Cardiac Magnetic Resonance Imaging (MRI)
#Corresponding Author: potato Team
#Affiliation: HuaQiao University
#Contact: [email/contact information]
#Version: v1.0.0
#Code Repository: [URL to code repository]
#Copyright © 2025 IEEE
#This work is licensed under the MIT License (see LICENSE for details)
#This code is intended exclusively for academic and research use.
#=====================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

def vectorize_fcn(X):
    """Vectorize functional connectivity matrices by extracting upper triangle."""
    n_samples, n_roi, _ = X.shape
    triu_indices = np.triu_indices(n_roi, k=1)
    X_vect = X[:, triu_indices[0], triu_indices[1]]  
    return X_vect

def main():
    # Load data and create labels
    X = np.load('significant_matrices.npy')
    y = np.array([1] * 33 + [0] * 50)  # 33 positive cases, 50 negative cases
    
    # Vectorize matrices
    X_vect = vectorize_fcn(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vect)
    
    # Apply diffusion maps for dimensionality reduction
    from pydiffmap import diffusion_map as dm
    diffusion = dm.DiffusionMap.from_sklearn(n_evecs=10, epsilon='bgh')
    X_dm = diffusion.fit_transform(X_scaled)
    
    # Define SVM parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    # Set up 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize lists to store performance metrics
    accuracies = []
    aucs = []
    sensitivities = []
    specificities = []
    
    fold = 1
    # Perform 5-fold cross-validation
    for train_idx, test_idx in cv.split(X_dm, y):
        print(f"Executing fold {fold} of cross-validation...")
        
        X_train, X_test = X_dm[train_idx], X_dm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Hyperparameter tuning with grid search
        svc = SVC(probability=True, random_state=42)
        grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        # Get best model and make predictions
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)  
        specificity = tn / (tn + fp)  
        
        # Store metrics
        accuracies.append(acc)
        aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
        fold += 1
    
    # Print summary of performance metrics
    print("\n5-fold cross-validation performance summary:")
    print(f"Average accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Average sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"Average specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")

if __name__ == '__main__':
    main()
