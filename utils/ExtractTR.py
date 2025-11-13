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

import os
import nibabel as nib
import pandas as pd

def get_tr_from_nifti(file_path):
    """Extract TR (Repetition Time) value from NIfTI file"""
    try:
        # Load NIfTI file
        img = nib.load(file_path)
        # Extract TR value (in seconds)
        tr = float(img.header.get_zooms()[-1])
        return tr
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def normalize_tr(tr):
    """Return 3.0 if TR value is within 0.001 of 3.0 (considered standard TR)"""
    if abs(tr - 3.0) <= 0.001:
        return 3.0
    return tr

# Set source folder path
source_dir = 'DataOut_with_ADCN'

# Create list to store results
results = []

# Traverse all files in the source folder and its subfolders
for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            file_path = os.path.join(root, filename)
            
            # Get TR value from NIfTI file
            tr = get_tr_from_nifti(file_path)
            
            if tr is not None:
                # Normalize TR value
                normalized_tr = normalize_tr(tr)
                
                # Get relative file path (relative to source directory)
                rel_path = os.path.relpath(file_path, source_dir)
                
                results.append({
                    'Filepath': rel_path,
                    'Original TR': tr,
                    'Normalized TR': normalized_tr
                })
                print(f"Processed: {rel_path}, TR = {normalized_tr} seconds")

# Create DataFrame and save to CSV files
if results:
    df = pd.DataFrame(results)
    
    # Save all TR values
    output_file = 'TR_values_all.csv'
    df.to_csv(output_file, index=False)
    print(f"\nAll TR values saved to {output_file}")
    
    # Save non-standard TR values (not equal to 3.0)
    abnormal_tr = df[df['Normalized TR'] != 3.0]
    if not abnormal_tr.empty:
        output_abnormal = 'TR_values_abnormal.csv'
        abnormal_tr.to_csv(output_abnormal, index=False)
        print(f"Non-standard TR values (!=3.0) saved to {output_abnormal}")
    
    # Display statistical summary
    print("\nTR Value Statistics:")
    print(f"Total files processed: {len(results)}")
    print(f"Files with standard TR (3.0): {len(df[df['Normalized TR'] == 3.0])}")
    print(f"Files with non-standard TR: {len(abnormal_tr)}")
    print(f"Average normalized TR: {df['Normalized TR'].mean():.3f} seconds")
    print(f"Minimum normalized TR: {df['Normalized TR'].min():.3f} seconds")
    print(f"Maximum normalized TR: {df['Normalized TR'].max():.3f} seconds")
else:
    print("No NIfTI files found or errors occurred during processing")
