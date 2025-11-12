# DFCN-SWDRC: 基于导数动态时间规整的功能连接网络阿尔茨海默病诊断

## 项目概述

本仓库实现了论文《Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks》（IEEE TRANSACTIONS ON MEDICAL IMAGING, 2020）中提出的方法，提供基于导数动态时间规整的功能连接网络构建和阿尔茨海默病诊断框架。

## 方法介绍

### 核心创新
DFCN-SWDRC（基于导数规律相关的滑动窗口动态功能连接网络）主要创新点：

1. **导数动态时间规整算法**：考虑脑区间信号传输的微分延迟
2. **滑动窗口技术**：结合滑动窗口提取脑活动的局部性和异步性特征
3. **功能延迟网络**：分析健康个体与AD患者脑区间信号传输的可度量延迟

### 算法流程图


## 核心算法

### 基于导数规律的相关性算法（CDR）

python
算法1：基于导数规律的相关性计算（CDR）
输入：X,Y-时间序列
输出：P_X'Y'-时间序列相关性
步骤：
1. 计算时间序列的导数变换
2. 基于导数序列计算距离矩阵
3. 使用动态时间规整计算路径
4. 路径回溯和序列对齐重建
5. 计算重建序列的Pearson相关系数


### DFCN-SWDRC构建算法

python
算法2：DFCN-SWDRC构建
输入：X,Y-时间序列，WindowSize-窗口大小，StepSize-步长
输出：CorrelationValue-时间序列相关性
步骤：
1. 对时间序列进行高斯平滑
2. 使用滑动窗口分割子序列
3. 对每个窗口应用CDR算法
4. 计算所有窗口的相关性均值


## 数据集

### 数据来源
- 使用ADNI数据库的88名受试者基线rs-fMRI数据
- 包含50名健康对照（HC）和38名AD患者
- 详细受试者信息如下：

**表I：参与者信息表**
| 类型 | AD | HC |
|------|-----|-----|
| 参与者数量 | 33 | 50 |
| 性别（男/女） | 16/17 | 22/28 |
| 平均年龄（标准差） | 72.5(7.2) | 75.4(6.6) |

### 数据预处理
使用MATLAB-based restplus工具包进行预处理：
- 去除前5个时间点
- 切片时间校正
- 头动校正（排除平移>2mm或旋转>2度的数据）
- 使用EPI模板进行空间标准化
- 协变量回归
- 6mm高斯核平滑
- 0.01-0.08Hz带通滤波
- 使用AAL模板将大脑划分为90个ROI

## 实验设置

### 超参数选择
通过消融实验确定最优滑动窗口参数：
- 窗口大小：30
- 步长：8
- 这些参数在top 1%功能连接的平均p值最小（0.002099）

### 网络特性分析

**小世界系数分析结果**


**聚类系数计算结果**


## 实验结果

### 显著差异功能连接
**表III：具有显著差异的功能连接**
| 连接的ROI（AAL编号） | p值(↓) |
|---------------------|---------|
| 中枕叶回(51)-下枕叶回(53) | 8.45×10⁻⁵ |
| 额上回眶部(5)-颞极：中颞叶回(88) | 1.141×10⁻⁴ |

**显著差异功能连接热图**


### 功能延迟网络分析

**显著差异脑区**
**表IV：具有显著差异的脑区**
| 脑区（AAL编号） | p值(↓) | 组间均值差异(↑) |
|----------------|---------|-----------------|
| 缘上回(63) | 1.098×10⁻⁶ | 2.289 |

**功能延迟网络差异示意图**


## 分类性能

### ADNI数据集二分类结果（HC vs. AD）
**表VI：ADNI数据集二分类结果**

| 方法 | 特征数量 | 准确率(↑) | AUC(↑) | 敏感度(↑) | 特异度(↑) |
|------|----------|-----------|---------|-----------|-----------|
| SFCN | 38 | 71.55% | 66.84% | 0% | 100% |
| FCN-SW | 329 | 86.74% | 94.42% | 78.78% | 92% |
| DFCN-SWDRC(我们的) | 295 | **96.38%** | **99.45%** | 93.93% | 98% |
| FDN(我们的) | 263 | 93.97% | 98.60% | 96.96% | 100% |

### ABIDE-UM数据集二分类结果（HC vs. ASD）
**表VII：ABIDE-UM数据集二分类结果**

| 方法 | 特征数量 | 准确率(↑) | AUC(↑) | 敏感度(↑) | 特异度(↑) |
|------|----------|-----------|---------|-----------|-----------|
| SFCN | 371 | 85.71% | 95.16% | 81.40% | 88.71% |
| DFCN-DRC(我们的) | 233 | **95.24%** | **99.10%** | 95.35% | 95.16% |
| FDN(我们的) | 197 | **95.24%** | 98.72% | 95.16% | 95.16% |

## 代码结构


.
├── data/                    # 数据目录
│   ├── adni/               # ADNI数据集
│   ├── abide/               # ABIDE数据集
│   └── preprocessed/        # 预处理后数据
├── src/                     # 源代码
│   ├── algorithms/          # 核心算法
│   │   ├── cdr.py          # CDR算法实现
│   │   ├── dtw.py          # 动态时间规整
│   │   └── sliding_window.py # 滑动窗口处理
│   ├── models/             # 模型定义
│   │   ├── fcn_models.py   # 功能连接网络模型
│   │   └── fdn_models.py   # 功能延迟网络模型
│   ├── utils/               # 工具函数
│   │   ├── preprocessing.py # 数据预处理
│   │   └── visualization.py # 可视化工具
│   └── evaluation/          # 评估模块
│       ├── classification.py # 分类评估
│       └── network_analysis.py # 网络分析
├── experiments/             # 实验脚本
│   ├── classification/      # 分类实验
│   ├── ablation/            # 消融实验
│   └── network_analysis/    # 网络分析实验
├── configs/                 # 配置文件
│   ├── data_config.yaml    # 数据配置
│   ├── model_config.yaml   # 模型配置
│   └── experiment_config.yaml # 实验配置
└── results/                 # 结果输出
    ├── figures/             # 生成图表
    └── tables/              # 结果表格


## 快速开始

### 环境配置

bash
conda create -n dfcn-swdrc python=3.8
conda activate dfcn-swdrc
pip install -r requirements.txt


### 数据预处理

python
from src.utils.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(config_path='configs/data_config.yaml')
preprocessed_data = preprocessor.process_raw_fmri(data_path='data/raw/')


### 功能连接网络构建

python
from src.algorithms.cdr import CDRAlgorithm
from src.models.fcn_models import DFCN_SWDRC

cdr = CDRAlgorithm()
dfcn_model = DFCN_SWDRC(cdr_algorithm=cdr, window_size=30, stride=8)
fcn_matrix = dfcn_model.build_network(time_series_data)


### 分类实验

bash
python experiments/classification/run_adni_classification.py \
  --config configs/experiment_config.yaml \
  --model dfcn-swdrc \
  --dataset adni


## 引用

如果您使用本代码或方法，请引用原始论文：

bibtex
@article{Hong2020,
  title={Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks},
  author={Xin Hong and Yongze Lin},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  volume={XX},
  number={XX},
  pages={1-11}
}


## 致谢

感谢ADNI和ABIDE项目提供的数据支持，以及所有为本研究做出贡献的研究人员。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 更新日志

### v1.0.0 (2024-12-01)
- 初始版本发布
- 实现DFCN-SWDRC核心算法
- 添加ADNI和ABIDE数据集支持
- 提供完整的实验复现脚本

### v0.1.0 (2024-11-15)
- 基础框架搭建
- 核心算法实现
- 初步实验验证

## 联系我们

如有问题或建议，请通过以下方式联系：
- 邮箱：xinhong@hqu.edu.cn
- GitHub Issues：[项目地址](https://github.com/your-repo/dfcn-swdrc)

---

**注意**: 本README文件根据原始论文内容整理，具体实现细节请参考源代码和原始论文。
