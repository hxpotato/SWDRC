# Alzheimer’s Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks
Official Repository of the paper: [Alzheimer’s Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks](https://github.com/hxpotato/SWDRC)

## Method
<p align="center">
  <img src="./figs/Fig. 2. The System Workflow.png" width="600" alt="FCN Construction">
</p>

<div align="center">
<b>Schematic Diagram of FCN Construction</b> (a) Static FCN; (b) DFCN Based on CDR
</div>

The study proposes three interconnected core techniques to address limitations of traditional functional connectivity analysis:
1. **CDR (Correlation based on Derivative Regularity)**: Alters time series features before similarity evaluation, uncovering lagged interactions overlooked by conventional methods. It aligns first-order signal derivatives via Dynamic Time Warping (DTW) to emphasize shape similarity and minimize temporal distortion.
2. **SWDRC (Sliding Window based on Derivative Regularity Correlation)**: Uses CDR in a sliding window format to align signal dynamics, capturing subtle spatio-temporal changes in neural regulation.
3. **FDN (Functional Delay Network)**: Measures relative transmission delays between brain regions, clarifying temporal patterns of signal propagation and quantifying interregional time offsets.


### Core Algorithm: CDR
The CDR algorithm proceeds through 6 key steps:
1. Derivative transformation of brain region time series to reflect local trend changes.
2. Distance matrix computation between derivative sequences of two brain regions.
3. Path computation using dynamic time warping to find minimum warping cost.
4. Path backtracking from sequence end to start to obtain optimal alignment.
5. Sequence alignment reconstruction based on optimal path index pairs.
6. Calculation of Pearson Correlation Coefficient (PCC) for reconstructed sequences.

### SWDRC Construction
1. Apply Gaussian smoothing to ROI time series.
2. Segment smoothed signals into overlapping subsequences using sliding window.
3. Use derivative Dynamic Time Warping (dDTW) within each window to synchronize subsequence derivatives.
4. Reconstruct aligned subsequences and compute PCC for each window.
5. Take mean PCC across all windows as edge weight for SWDRC network.

### FDN Construction
1. Record optimal matching path within each sliding window during SWDRC construction.
2. Calculate average index difference of path pairs across all windows to quantify interregional delays.
3. Aggregate delay coefficients for all ROI pairs to build FDN.

## Dataset
Two public datasets were used for experiments:

### ADNI Dataset
- 417 subjects: 165 Healthy Controls (HC), 108 Alzheimer’s Disease (AD) patients, 144 Mild Cognitive Impairment (MCI) individuals.
- Mean age: AD (75.11±5.92), HC (74.88±7.65), MCI (72.26±7.30).
- Imaging parameters: 3.0T MRI scanner, EPI sequences, 140 time points, TE=30ms, TR=3000ms, 48 slices, spatial resolution=3.31×3.31×3.31mm³, FA=80°.

### ABIDE Dataset
- Data from two largest sites (NYU, UM): 116 Autism Spectrum Disorder (ASD) patients, 156 HC.
- Mean age: ASD (14.51±5.81), HC (15.48±5.42).
- Standardized preprocessing including head motion correction, temporal smoothing, spatial normalization, and denoising.

### Data Preprocessing
Performed using MATLAB-based RESTplus toolkit:
1. Removal of first 5 time points.
2. Slice-timing correction and head motion correction (excluding data with translations >2mm or rotations >2°).
3. Spatial normalization to EPI template.
4. Regression of nuisance covariates (6 head motion parameters, white matter signals, cerebrospinal fluid signals, global brain signals).
5. Smoothing with 6mm Gaussian kernel and band-pass filtering (0.01–0.08Hz).
6. Parcellation into 90 ROIs using AAL template and extraction of mean time series.

## Code Structure
```
.
├── configs                   # Experiment configuration files
├── data                      # Dataset storage and preprocessing scripts
│   ├── adni_preprocess.py    # ADNI dataset preprocessing
│   ├── abide_preprocess.py   # ABIDE dataset preprocessing
│   └── utils.py              # Data utility functions
├── models                    # Model definitions
│   ├── cdr.py                # CDR algorithm implementation
│   ├── swdrc.py              # SWDRC network construction
│   ├── fdn.py                # FDN network construction
│   └── utils.py              # Model utility functions
├── experiments               # Experiment scripts
│   ├── classification.py     # Classification task implementation
│   ├── network_analysis.py   # Network property analysis
│   └── hyperparameter_search.py # Grid search for optimal parameters
├── utils                     # Shared utilities
│   ├── metrics.py            # Evaluation metrics (ACC, AUC, sensitivity, specificity)
│   ├── visualization.py      # Result visualization tools
│   └── tools.py              # General helper functions
├── main.py                   # Main entry point for running experiments
└── requirements.txt          # Dependencies
```

## Key Experimental Results
### Hyperparameter Optimization
- Optimal sliding window size: 10, step size: 4 (minimized mean p-value of top 1% functional connections).

### Network Properties
- **Small-World Coefficient**: SWDRC network shows stronger small-world characteristics than comparative methods (SFCN, FCN-SW, DCC, etc.) across sparsity 0.1–0.4.
- **Clustering Coefficient**: SWDRC achieves higher clustering coefficients at all sparsity levels, capturing modular structure of brain functions.

### Classification Performance
| Task | Method | Accuracy (%) | AUC (%) |
|------|--------|--------------|---------|
| HC vs. AD (ADNI) | SWDRC+FDN | 90.39±0.65 | 96.31±0.61 |
| HC vs. MCI (ADNI) | SWDRC+FDN | 84.99±0.66 | 88.43±0.73 |
| HC vs. MCI vs. AD (ADNI) | SWDRC+FDN | 86.49±0.62 | 91.12±0.48 |
| HC vs. ASD (ABIDE) | SWDRC+FDN | 92.47±0.58 | 96.91±0.52 |

### FDN Analysis
- Network delay increases from HC → MCI → AD, with significant differences in key brain regions (lenticular nucleus, supplementary motor area, superior occipital gyrus).
- Critical delayed connections: Posterior cingulate cortex-angular gyrus (AD vs. HC), orbitofrontal cortex-temporal pole (MCI vs. HC).

## Launching Experiments
### Environment
```
conda create -n ad_diagnosis python=3.8
pip install -r requirements.txt
```

### Run Classification Experiments
```
python main.py \
  --task classification \
  --dataset adni \
  --model swdrc+fdn \
  --window_size 10 \
  --step_size 4 \
  --classifier svm
```

### Run Network Analysis
```
python main.py \
  --task network_analysis \
  --dataset adni \
  --model swdrc \
  --sparsity 0.2
```

## Acknowledgement
This work is supported by:
- Scientific Research Start-up Fund Project for High-level Researchers of Huaqiao University (Grant 22BS105)
- Scientific and Technological Major Special Project of Fujian Provincial Health Commission (Grants 2023Y9376, 2021ZD01004)
- Natural Science Foundation of Fujian Province, China (Grant 2022J01318)

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{hong2025alzheimer,
  title={Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks},
  author={Hong, Xin and Lin, Yongze and Wu, Zhenghao},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

