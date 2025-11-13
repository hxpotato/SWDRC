DFCN-SWDRC: Alzheimer's Disease Diagnosis Using Derivative-based Dynamic Time Warping Functional Connectivity Networks

Project Overview

This repository implements the method proposed in the paper "Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks" (IEEE TRANSACTIONS ON MEDICAL IMAGING, 2020). It provides a framework for constructing functional connectivity networks using derivative dynamic time warping and diagnosing Alzheimer's Disease.

Method Introduction

Core Innovations

DFCN-SWDRC (Derivative-based Functional Connectivity Network with Sliding Window and Dynamic Time Warping Correlation) main innovations:

1. Derivative Dynamic Time Warping Algorithm: Considers the differential delays in signal transmission between brain regions.
2. Sliding Window Technique: Incorporates sliding windows to extract local and asynchronous features of brain activity.
3. Functional Delay Network: Analyzes measurable delays in signal transmission between brain regions in healthy individuals and AD patients.

Algorithm Flowchart

(Flowchart would be inserted here)

Core Algorithms

Correlation based on Derivative Rules (CDR) Algorithm

Algorithm 1: Correlation Calculation based on Derivative Rules (CDR)
Input: X, Y - time series
Output: P_X'Y' - time series correlation
Steps:
1. Calculate derivative transformation of time series
2. Compute distance matrix based on derivative sequences
3. Calculate path using dynamic time warping
4. Perform path backtracking and sequence alignment reconstruction
5. Compute Pearson correlation coefficient of reconstructed sequences


DFCN-SWDRC Construction Algorithm

Algorithm 2: DFCN-SWDRC Construction
Input: X, Y - time series, WindowSize - window size, StepSize - step size
Output: CorrelationValue - time series correlation
Steps:
1. Apply Gaussian smoothing to time series
2. Split into subsequences using sliding windows
3. Apply CDR algorithm to each window
4. Calculate mean correlation across all windows


Dataset

Data Source

• Uses baseline rs-fMRI data from 88 subjects in the ADNI database

• Includes 50 healthy controls (HC) and 38 AD patients

• Detailed participant information:

Table I: Participant Information
Group AD HC

Number of Participants 33 50

Gender (Male/Female) 16/17 22/28

Mean Age (Standard Deviation) 72.5(7.2) 75.4(6.6)

Data Preprocessing

Preprocessing using MATLAB-based restplus toolkit:
• Remove first 5 time points

• Slice timing correction

• Head motion correction (excluding data with >2mm translation or >2° rotation)

• Spatial normalization using EPI template

• Covariate regression

• Smoothing with 6mm Gaussian kernel

• Band-pass filtering (0.01-0.08Hz)

• Brain parcellation into 90 ROIs using AAL template

Experimental Setup

Hyperparameter Selection

Optimal sliding window parameters determined through ablation experiments:
• Window size: 30

• Step size: 8

• These parameters yielded the smallest average p-value (0.002099) for top 1% functional connections

Network Property Analysis

Small-World Coefficient Analysis Results
(Results would be inserted here)

Clustering Coefficient Calculation Results
(Results would be inserted here)

Experimental Results

Significant Differential Functional Connections

Table III: Functional Connections with Significant Differences
Connected ROIs (AAL Number) p-value(↓)

Middle Occipital Gyrus(51)-Inferior Occipital Gyrus(53) 8.45×10⁻⁵⁵

Superior Frontal Gyrus, Orbital(5)-Temporal Pole: Middle Temporal Gyrus(88) 1.141×10⁻⁴⁴

Heatmap of Significant Differential Functional Connections
(Heatmap would be inserted here)

Functional Delay Network Analysis

Significant Differential Brain Regions
Table IV: Brain Regions with Significant Differences
Brain Region (AAL Number) p-value(↓) Mean Difference Between Groups(↑)

Supramarginal Gyrus(63) 1.098×10⁻⁶⁶ 2.289

Schematic Diagram of Functional Delay Network Differences
(Diagram would be inserted here)

Classification Performance

Binary Classification Results on ADNI Dataset (HC vs. AD)

Table VI: Binary Classification Results on ADNI Dataset

Method Number of Features Accuracy(↑) AUC(↑) Sensitivity(↑) Specificity(↑)

SFCN 38 71.55% 66.84% 0% 100%

FCN-SW 329 86.74% 94.42% 78.78% 92%

DFCN-SWDRC (Ours) 295 96.38% 99.45% 93.93% 98%

FDN (Ours) 263 93.97% 98.60% 96.96% 100%

Binary Classification Results on ABIDE-UM Dataset (HC vs. ASD)

Table VII: Binary Classification Results on ABIDE-UM Dataset

Method Number of Features Accuracy(↑) AUC(↑) Sensitivity(↑) Specificity(↑)

SFCN 371 85.71% 95.16% 81.40% 88.71%

DFCN-DRC (Ours) 233 95.24% 99.10% 95.35% 95.16%

FDN (Ours) 197 95.24% 98.72% 95.16% 95.16%

Code Structure


.
├── data/                    # Data directory
│   ├── adni/               # ADNI dataset
│   ├── abide/              # ABIDE dataset
│   └── preprocessed/       # Preprocessed data
├── src/                    # Source code
│   ├── algorithms/         # Core algorithms
│   │   ├── cdr.py         # CDR algorithm implementation
│   │   ├── dtw.py         # Dynamic Time Warping
│   │   └── sliding_window.py # Sliding window processing
│   ├── models/            # Model definitions
│   │   ├── fcn_models.py  # Functional connectivity network models
│   │   └── fdn_models.py  # Functional delay network models
│   ├── utils/             # Utility functions
│   │   ├── preprocessing.py # Data preprocessing
│   │   └── visualization.py # Visualization tools
│   └── evaluation/        # Evaluation modules
│       ├── classification.py # Classification evaluation
│       └── network_analysis.py # Network analysis
├── experiments/           # Experiment scripts
│   ├── classification/    # Classification experiments
│   ├── ablation/          # Ablation experiments
│   └── network_analysis/ # Network analysis experiments
├── configs/              # Configuration files
│   ├── data_config.yaml  # Data configuration
│   ├── model_config.yaml # Model configuration
│   └── experiment_config.yaml # Experiment configuration
└── results/              # Result outputs
    ├── figures/          # Generated figures
    └── tables/           # Result tables


Quick Start

Environment Setup

conda create -n dfcn-swdrc python=3.8
conda activate dfcn-swdrc
pip install -r requirements.txt


Data Preprocessing

from src.utils.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(config_path='configs/data_config.yaml')
preprocessed_data = preprocessor.process_raw_fmri(data_path='data/raw/')


Functional Connectivity Network Construction

from src.algorithms.cdr import CDRAlgorithm
from src.models.fcn_models import DFCN_SWDRC

cdr = CDRAlgorithm()
dfcn_model = DFCN_SWDRC(cdr_algorithm=cdr, window_size=30, stride=8)
fcn_matrix = dfcn_model.build_network(time_series_data)


Classification Experiment

python experiments/classification/run_adni_classification.py \
  --config configs/experiment_config.yaml \
  --model dfcn-swdrc \
  --dataset adni


Citation

If you use this code or method, please cite the original paper:
@article{Hong2020,
  title={Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks},
  author={Xin Hong and Yongze Lin},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  volume={XX},
  number={XX},
  pages={1-11}
}


Acknowledgments

Thanks to the ADNI and ABIDE projects for data support, and all researchers who contributed to this study.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Changelog

v1.0.0 (2024-12-01)

• Initial version release

• Implementation of DFCN-SWDRC core algorithms

• Added support for ADNI and ABIDE datasets

• Provided complete experiment reproduction scripts

v0.1.0 (2024-11-15)

• Basic framework setup

• Core algorithm implementation

• Preliminary experimental validation

Contact Us

For questions or suggestions, please contact:
• Email: xinhong@hqu.edu.cn

• GitHub Issues: https://github.com/your-repo/dfcn-swdrc

Note: This README is compiled based on the content of the original paper. For specific implementation details, please refer to the source code and the original paper.
