#-- coding: utf-8 -*-
import os
import numpy as np
from dtaidistance import dtw
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ====== 用户配置区 ======
input_folder  = "G:/WPSData/Code/TMI_lastexp/roi_data/data_EMCI_CN"   # 原始 ROI 时间序列 txt 所在文件夹
output_folder = "G:/WPSData/Code/TMI_lastexp/out_put_all/ori_dtw/EMCI_CN"  # DTW 矩阵要保存到哪个文件夹
OUTPUT_FOLDER = output_folder
INPUT_FOLDER = input_folder
N_ROIS        = 90                             # AAL90 模板的 ROI 数量
MAX_WORKERS   = multiprocessing.cpu_count()    # 并行进程数，默认为 CPU 核数
# ========================

def process_file(fname):
    """读取单个 txt 文件，截取前 N_ROIS 列，计算 90×90 的原始 DTW 距离矩阵，保存输出。"""
    in_path  = os.path.join(INPUT_FOLDER, fname)
    out_path = os.path.join(OUTPUT_FOLDER, fname)

    # 1) 载入并截取前 N_ROIS 列
    data = np.loadtxt(in_path)
    if data.shape[1] < N_ROIS:
        raise ValueError(f"{fname} 列数不足 {N_ROIS} 列，无法截取 AAL90。")
    data = data[:, :N_ROIS]  # 形状 (T, N_ROIS)

    # 2) 初始化输出矩阵
    dtw_mat = np.zeros((N_ROIS, N_ROIS), dtype=float)

    # 3) 两两计算 DTW 距离（原始实现）
    for i in range(N_ROIS):
        sig_i = data[:, i]
        for j in range(i, N_ROIS):
            sig_j = data[:, j]
            dist = dtw.distance(sig_i, sig_j)
            dtw_mat[i, j] = dist
            dtw_mat[j, i] = dist

    # 4) 保存结果
    np.savetxt(out_path, dtw_mat, fmt="%.6f")
    print(f"✔ {fname} ➞ DTW 矩阵保存到 {out_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    # 收集所有待处理的 .txt 文件名
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")]
    print(MAX_WORKERS)
    # 并行分发：每个进程跑一个文件
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_file, files)