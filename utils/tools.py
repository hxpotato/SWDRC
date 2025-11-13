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
from scipy.stats import ttest_ind
import pandas as pd
import json
import warnings
from scipy.stats import ttest_ind
from scipy.stats import permutation_test
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
output_file_path = 
def extract_all_metrics(data):
    all_metrics_df = pd.DataFrame()
    for col in data.columns[:-1]:  
        node_df = pd.DataFrame()
        for index, cell in data[col].iteritems():
            if pd.notnull(cell):
                cell_dict = json.loads(cell.replace("'", "\""))
                cell_series = pd.Series(cell_dict)
                node_df = node_df.append(cell_series, ignore_index=True)
            else:
                node_df = node_df.append(pd.Series([None] * len(node_df.columns)), ignore_index=True)
        node_df['node'] = col
        all_metrics_df = pd.concat([all_metrics_df, node_df], ignore_index=True)
    return all_metrics_df
def main(test_AD_file_path, test_CN_file_path, output_file_path):
    ad_data = pd.read_csv(test_AD_file_path)
    cn_data = pd.read_csv(test_CN_file_path)
    ad_all_metrics_df = extract_all_metrics(ad_data)
    cn_all_metrics_df = extract_all_metrics(cn_data)
    results_list = []  
    for node_to_test in ad_all_metrics_df['node'].unique():
        ad_node_data = ad_all_metrics_df[ad_all_metrics_df['node'] == node_to_test].drop(columns=['node'])
        cn_node_data = cn_all_metrics_df[cn_all_metrics_df['node'] == node_to_test].drop(columns=['node'])
        metrics = ad_node_data.columns
        for metric in metrics:
            ad_metric_data = ad_node_data[metric].dropna()
            cn_metric_data = cn_node_data[metric].dropna()
            t_stat, p_value = ttest_ind(ad_metric_data, cn_metric_data, equal_var=False)  
            results_list.append({
                'Node': node_to_test,
                'Metric': metric,
                'T-statistic': t_stat,
                'P-value': p_value
            })
            if(p_value<0.0001):
                print(f"Node {node_to_test} with metric {metric} has p-value {p_value}")
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='P-value')
        results_df.to_csv(output_file_path, index=False)
def out_main():
    for w in range(10, 55, 5):
        for s in range(1, 11):
            print(f"Processing {w}_{s}...")
            AD_file_path = 
            CN_file_path = 
            output_file_path = 
            main(AD_file_path, CN_file_path,output_file_path)
            print(f"Results for {w}_{s} have been saved to DMN_t_test_results.csv.")
def out_one_main():
    prefix = 
    AD_file_path = prefix
    CN_file_path = prefix+
    output_file_path = prefix+
    main(AD_file_path, CN_file_path,output_file_path)
if __name__ == '__main__':
    out_one_main()
    print("all metrics extracted successfully!")
def unuse():
    print("all metrics extracted successfully!")
