import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


in_folder='./ADCN_ROI/'
# out_folder='H:/DtwData/dDTW_Ori/per/'

out_folder='H:/DtwData/dDTW_Ori_sqrt/per/'
def gaussian_kernel(size, sigma):
    return np.exp(-(np.arange(size) - size // 2) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussian_smooth(data, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel /= np.sum(kernel)  # 归一化核
    # 创建镜像边界
    extended_data = np.pad(data, pad_width=kernel_size//2, mode='reflect')
    smoothed_data = convolve(extended_data, kernel, mode='valid')
    return smoothed_data


def load_one_file(file_path):
    matrix = np.loadtxt(file_path)
    transposed_matrix = np.transpose(matrix)
    return transposed_matrix

def compute_derivative(series):
    # 计算一阶导数
    Dq = np.zeros_like(series)
    for i in range(1, len(series) - 1):
        Dq[i] = (series[i] - series[i - 1] + (series[i + 1] - series[i - 1]) / 2) / 2
    Dq[0] = Dq[1]  # 边界处理
    Dq[-1] = Dq[-2]  # 边界处理
    return Dq

def cal_dis_mat(row1,row2,cols):
    '''
    计算两行之间的dis矩阵
    :param cols: 行的长度
    返回0-(cols-1)的下标的矩阵
    '''
    #创建空矩阵
    A_deriv = compute_derivative(row1)
    B_deriv = compute_derivative(row2)
    dis = np.zeros((cols, cols))
    for li in range(0,cols):
        for lj in range(0,cols):
            # 计算导数之间的距离矩阵
            # dis[li, lj] = (A_deriv[li] - B_deriv[lj]) ** 2
            dis[li, lj] = np.sqrt((A_deriv[li] - B_deriv[lj]) ** 2)
    return dis

def dtw_from_distance_matrix(dis_mat):
    '''
    计算DTW矩阵，该矩阵是从位置（1，1）开始，才有实际上的意义，(0，0)处置为0,代表两个序列都从其开始位置（没有任何先前的数据）开始时的累积成本。
    :param dis_mat:距离矩阵，从0开始
    :return:
    '''
    n, m = dis_mat.shape
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    #注意，在i=0或者j=0处，全为无穷大，因此当（u=1，v>1时）或者当（u>1，v=1时）
    #不会从u=0或者v=0处转移过来
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dis_mat[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix

def backtrack_dtw(dtw_matrix):
    '''
    找到使两个序列之间的总距离最小的路径。
    '''
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    path = [(i-1, j-1)]
    while i > 1 or j > 1:
        #当i == 1时，只能移动j//当j == 1时，只能移动i
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        #当i>1,j>1时，有三种移动方式，选择其中最小的（因为是从最小的方向转移过来的）
        else:
            #向上的值是三个方向中最小的，那么选择向上
            if dtw_matrix[i-1, j] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                i -= 1
            #如果向左的值是三个方向中最小的，那么选择向左
            elif dtw_matrix[i, j-1] == min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]):
                j -= 1
            #如果向左上的值是三个方向中最小的，那么选择向左上
            else:
                i -= 1
                j -= 1
        path.append((i-1, j-1))
    path.reverse()
    return path


def cal_D(ormat, n):
    '''
    计算原始信号矩阵的各个信号之间的规整前后的皮尔森相关系数
    :param ormat: 原始信号矩阵
    :param n: 信号个数
    '''
    # ##  ！！！注意，可以尝试不同的标准化方法，比如z-score标准化等
    # #计算每行总值
    # row_means = ormat.sum(axis=1, keepdims=True)
    # # 每个元素除以其行的总值，进行标准化
    # normalized_matrix = ormat / row_means

    # 高斯平滑
    kernel_size = 9
    # kernel_size = 5
    # kernel_size需要为奇数，sigma为标准差
    sigma = 2
    for i in range(n):
        data = ormat[i]
        smoothed_data = gaussian_smooth(data, kernel_size, sigma)
        ormat[i] = smoothed_data
    normalized_matrix = ormat


    cols = len(ormat[0])
    retp = np.zeros((n, n))
    #计算规整路径矩阵D
    #对第i行，第j行
    for ri in range(0,n):
        for rj in range(0,n):
            if ri==rj:
                retp[ri][rj]=1
                continue
            if ri>rj:
                continue
            else :
                #计算两行之间的dis矩阵
                dis_mat = cal_dis_mat(normalized_matrix[ri],normalized_matrix[rj],cols)
                #进行DTW计算
                dtw=dtw_from_distance_matrix(dis_mat)
                #还原路径
                path = backtrack_dtw(dtw)

                aligned_a, aligned_b = warp_sequences(normalized_matrix[ri],normalized_matrix[rj],path)
                newp=pearson_correlation(aligned_a, aligned_b)
                retp[ri][rj] = newp
                retp[rj][ri] = newp
    return retp



def warp_sequences(seq_a, seq_b, path):
    '''
    还原规整后的序列
    '''
    aligned_a = []
    aligned_b = []

    for (i, j) in path:
        aligned_a.append(seq_a[i])  # !!!不能-1！减1是因为path是基于dtw_matrix的，它比seq_a和seq_b大1
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b


def pearson_correlation(seq1, seq2):
    """计算两个序列之间的皮尔森相关系数"""
    if(np.std(seq1)==0 or np.std(seq1)==0):
        print("存在标准差为0的序列！")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    np.savetxt(output_path, matrix,delimiter=',')

def main(in_folder,out_folder):
    for filename in os.listdir(in_folder):
        if filename.endswith(".txt"):
            # 输入输出文件路径的定义，矩阵的读入（已经转置）
            in_file_path = os.path.join(in_folder, filename)
            outp_file_path = os.path.join(out_folder , filename)
            matrix = load_one_file(in_file_path)
            # 计算规整后的皮尔森相关系数矩阵，并存储
            retp = cal_D(matrix,90)
            save_mat_to_txt(retp,outp_file_path)

def process_one_file(filename, in_folder, out_folder):
    # 输入输出文件路径的定义，矩阵的读入（已经转置）
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)
    # 计算规整后的皮尔森相关系数矩阵，并存储
    retp = cal_D(matrix, 90)
    save_mat_to_txt(retp, outp_file_path)


def multi_main():
    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(in_folder), desc="处理文件进度"):
        if filename.endswith(".txt"):
            # 直接调用处理函数
            process_one_file(filename, in_folder, out_folder)

if __name__ == '__main__':
    multi_main()