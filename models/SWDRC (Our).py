import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

in_folder='./ROIsingal'
MA_folder='./Per/new/MAC2'

def load_one_file(file_path):
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def save_mat_to_txt(matrix, output_path):
    np.savetxt(output_path, matrix,delimiter=' ')

def pearson_correlation(seq1, seq2):
    """计算两个序列之间的皮尔森相关系数"""
    if(np.std(seq1)==0 or np.std(seq1)==0):
        print("存在标准差为0的序列！")
    return np.corrcoef(seq1, seq2)[0, 1]
def sliding_window(seq1, seq2, window_size=10, stride=1):
    """
    计算滑动窗口内的两序列之间的所有滑动窗口序列对
    参数:
    :param seq1: 第一个序列。
    :param seq2: 第二个序列。
    :param window_size: 滑动窗口的大小。
    :param stride: 每次迭代窗口移动的位置数量。

    返回:
    :return: 一个元组列表，其中每个元组包含每个窗口的两个对齐序列。
    """

    if len(seq1) != len(seq2):
        raise ValueError("Input sequences must have the same length.")

    all_aligned_pairs = []

    for start in range(0, len(seq1) - window_size + 1, stride):
        end = start + window_size

        tseq1 = seq1[start:end]
        tseq2 = seq2[start:end]
        all_aligned_pairs.append((tseq1, tseq2))

    return all_aligned_pairs


def cal_window(ormat, n, window_size=10, stride=1):
    """
    计算滑动窗口内的两序列的平均皮尔森相关系数
    参数:
    :param seq1: 第一个序列。
    :param seq2: 第二个序列。
    :param window_size: 滑动窗口的大小。
    :param stride: 每次迭代窗口移动的位置数量。
    返回:
    :return: 一个元组列表，其中每个元组包含每个窗口的两个对齐序列。
    """
    retp = np.zeros((n, n))
    for ri in range(0, n):
        for rj in range(ri+1, n):  # 注意这里，我们从ri+1开始，而不是从0开始
            #传入全长序列
            aligned_pairs = sliding_window(ormat[ri], ormat[rj], window_size, stride)
            local_pearsons = [pearson_correlation(pair[0], pair[1]) for pair in aligned_pairs]
            avg_pearson = np.mean(local_pearsons)
            retp[ri][rj] = avg_pearson
            retp[rj][ri] = avg_pearson  # 利用皮尔森系数的对称性，减少计算量
    for ri in range(0, n):
        retp[ri][ri] = 1
    return retp

def main():
    for filename in tqdm(os.listdir(in_folder), desc="Processing files",position=0,leave=True):
        if filename.endswith(".txt"):
            in_file_path = os.path.join(in_folder, filename)
            MA_file_path = os.path.join(MA_folder , 'Per'+filename)
            matrix = load_one_file(in_file_path)
            ret_s_DTW = cal_window(matrix,90,10,1)
            save_mat_to_txt(ret_s_DTW, MA_file_path)

if __name__ == '__main__':
    main()