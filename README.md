Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks

[!NOTE]

Official Repository of the paper: 文档13链接 (IEEE Transactions on Medical Imaging, 2025)

Method

<div align="center">
  
</div>
<div align="center">
  <b>Fig. 2. The System Workflow.</b> Constructing CDR, SWDRC, and FDN for Alzheimer's disease diagnosis.
</div>

This work introduces a novel framework for analyzing brain functional connectivity by addressing the dynamic changes and complex time-delay characteristics of signal transmission between brain regions. The core innovation lies in three interconnected techniques:

1.  Correlation based on Derivative Regularity (CDR): This algorithm performs non-linear alignment of fMRI time series derivatives using Dynamic Time Warping (DTW), enabling the capture of transient, physiologically relevant lagged interactions that conventional correlation methods might overlook.
2.  Sliding Window based on Derivative Regularity Correlation (SWDRC): A dynamic functional connectivity network (DFCN) construction method that applies CDR within a sliding window framework to identify local and asynchronous characteristics of brain activity.
3.  Functional Delay Network (FDN): This network explicitly quantifies the relative signal transmission delays between different brain regions, providing a novel perspective on the temporal patterns of information propagation in the brain.

The synergy of SWDRC and FDN allows the model to capture fine-grained temporal alignment features while reflecting explicit delay information, enhancing the discrimination between healthy controls and patients with Alzheimer's disease (AD) or Mild Cognitive Impairment (MCI).

Dataset

The research utilizes two primary datasets:

1.  ADNI Dataset: Resting-state fMRI data from 417 subjects, including 165 Healthy Controls (HC), 108 patients with Alzheimer's Disease (AD), and 144 individuals with Mild Cognitive Impairment (MCI).
2.  ABIDE Dataset: Data from the NYU and UM sites are used for additional validation, comprising 116 individuals with Autism Spectrum Disorder (ASD) and 156 Healthy Controls (HC).

Data preprocessing was performed using the MATLAB-based RESTplus toolkit, including steps like slice-timing correction, head motion correction, spatial normalization, and band-pass filtering. The brain was parcellated into 90 Regions of Interest (ROIs) using the AAL template.

Code Structure

The provided code and models are available at: https://github.com/hxpotato/SWDRC

(Note: The specific code structure details from the PDF are limited. The repository link above is the primary reference for implementation.)

Experimental Results & Performance

The proposed methods were rigorously evaluated against several state-of-the-art static and dynamic functional connectivity network construction methods.

Key Findings:

•   Network Properties: The SWDRC network demonstrated distinctly stronger small-world characteristics compared to other methods, closely aligning with the efficient information transmission observed in real human brain networks.

    
    <div align="center">
      <b>Fig. 6. Small-World Coefficient Calculation Results.</b> SWDRC shows superior small-world properties.
    </div>

•   Superior Classification Performance: The combination of SWDRC and FDN (SWDRC+FDN) achieved the highest classification accuracy in distinguishing HC from AD, HC from MCI, and in three-class classification (HC vs. MCI vs. AD) on the ADNI dataset. It also demonstrated robust performance on the ABIDE dataset (HC vs. ASD).

<table>
    <tr>
        <td colspan="7"><b>Table IV: Classification Results on ADNI (HC vs. AD) using SVM</b></td>
    </tr>
    <tr>
        <td><b>Method</b></td>
        <td><b>Network Type</b></td>
        <td><b>Accuracy(%)↑</b></td>
        <td><b>AUC(%)↑</b></td>
        <td><b>Sensitivity(%)↑</b></td>
        <td><b>Specificity(%)↑</b></td>
    </tr>
    <tr>
        <td>SFCN[28]</td>
        <td>Static</td>
        <td>85.72±0.42</td>
        <td>91.30±0.68</td>
        <td>79.10±0.95</td>
        <td>89.72±0.53*</td>
    </tr>
    <tr>
        <td>FCN-SW[43]</td>
        <td>Dynamic</td>
        <td>85.27±0.63</td>
        <td>91.81±0.74</td>
        <td>79.20±0.88</td>
        <td>89.12±0.47</td>
    </tr>
    <tr>
        <td>Brain-JEPA[26]</td>
        <td>Dynamic</td>
        <td>88.62±0.58*</td>
        <td>93.12±0.74</td>
        <td>84.64±0.89*</td>
        <td>89.16±0.66</td>
    </tr>
    <tr>
        <td><b>SWDRC+FDN (Ours)</b></td>
        <td><b>Dynamic</b></td>
        <td><b>90.39±0.65</b></td>
        <td><b>96.31±0.61</b></td>
        <td><b>88.49±0.72</b></td>
        <td><b>91.52±0.58</b></td>
    </tr>
</table>

•   Functional Delay Network Analysis: The FDN revealed pronounced alterations in signal transmission delays in AD and MCI groups compared to HC, particularly within the default mode network and connections involving frontal and sensory-related regions. This provides insights into the pathological mechanisms of AD.

    
    <div align="center">
      <b>Fig. 8. Schematic Diagram of FDN Differences (HC vs. AD).</b> Thicker edges indicate longer delays in the AD group.
    </div>

Conclusion

This study addresses the limitations of traditional functional connectivity methods by accounting for the asynchronous nature of interregional brain communication. The SWDRC and FDN frameworks, built upon the novel CDR algorithm, effectively capture dynamic and delay-sensitive features from fMRI data. The combined approach (SWDRC+FDN) demonstrates superior performance in classifying neurodegenerative conditions, offering a powerful and interpretable tool for early diagnosis and understanding of brain network alterations in diseases like Alzheimer's.

Citation

If you find this repository useful in your research, please consider citing the original paper:

@article{hong2025alzheimer,
  title={Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks},
  author={Hong, Xin and Lin, Yongze and Wu, Zhenghao},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
