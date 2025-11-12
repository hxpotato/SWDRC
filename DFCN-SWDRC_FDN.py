import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
import numpy as np
import os


in_folder='./ADCN_ROI/'
out_folder='H:/DtwData/dDTW_Ori_sqrt_new/WDO_per1111/'

window_size = [i for i in range(10, 55, 5)]
#原先平方均已完成，现在计算平方根下的结果
stride = [i for i in range(1, 11)]
# stride = [i for i in range(1, 5)]已完成
# stride = [i for i in range(5, 9)]已完成
# stride = [i for i in range(9, 11)]已完成

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
    # 注意，路径在此时已经-1，无需在warp_sequences中再-1
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
def sliding_window_DTW(seq1, seq2, window_size=10, stride=1):
    """
    计算滑动窗口内的两序列之间的DTW，并返回每个窗口的对齐序列。

    参数:
    :param seq1: 第一个序列。
    :param seq2: 第二个序列。
    :param window_size: 滑动窗口的大小。
    :param stride: 每次迭代窗口移动的位置数量。
    :param path_cal: 记录规整路径的匹配次数
    返回:
    :return: 一个元组列表，其中每个元组包含每个窗口的两个对齐序列。
    """
    all_aligned_pairs = []
    # 若是存储序列对齐路径，则取消注释
    # paths = []
    # 初始化规整路径计算矩阵
    path_cal = np.zeros((len(seq1)+5, len(seq2)+5))

    for start in range(0, len(seq1) - window_size + 1, stride):
        end = start + window_size

        normalized_seq1 = seq1[start:end]

        normalized_seq2 = seq2[start:end]

        dis_mat = cal_dis_mat(normalized_seq1, normalized_seq2, end - start)
        dtw_matrix = dtw_from_distance_matrix(dis_mat)
        path = backtrack_dtw(dtw_matrix)

        for (i, j) in path:
            path_cal[i, j] += 1
        #paths.append(path)
        aligned_a, aligned_b = warp_sequences(normalized_seq1, normalized_seq2, path)
        all_aligned_pairs.append((aligned_a, aligned_b))

    # return all_aligned_pairs, paths
    return all_aligned_pairs, path_cal

def cal_window_D(ormat, n, window_size, stride,path_cal_save_folder,patient_id):
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
    # 先对全序列做归一化处理

    # cols = len(ormat[0])
    retp = np.zeros((n, n))
    # path_cal矩阵记录规整路径的匹配次数
    ALL_ROI_path_cal = np.zeros((n, n))
    #  ！！！！注意，需要在里面记录，因为是对于不同的ROI之间的计算
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
                # 传入全长序列
                aligned_pairs , t_path_cal = sliding_window_DTW(ormat[ri], ormat[rj], window_size, stride)
                # 根据所有的对齐序列计算皮尔森相关系数
                local_pearsons = [pearson_correlation(pair[0], pair[1]) for pair in aligned_pairs]
                # 计算并保存平均皮尔森相关系数
                retp[ri][rj] = np.mean(local_pearsons)
                retp[rj][ri] = retp[ri][rj]

                # 计算规整路径的匹配次数
                path_cal_sum = np.sum(t_path_cal)
                # 计算行和列对应的和
                ri_cal = 0.0
                rj_cal = 0.0
                sqe_len = len(ormat[ri])
                # 注意：path为(0,0),(0,1),(1,2)...(n,n)
                for tmp_i in range(sqe_len+3):
                    for tmp_j in range(sqe_len+3):
                        ri_cal = ri_cal + (tmp_i * t_path_cal[tmp_i][tmp_j])
                        rj_cal = rj_cal + (tmp_j * t_path_cal[tmp_i][tmp_j])

                # 计算平均值差值
                path_val = (ri_cal - rj_cal) / (path_cal_sum)
                ALL_ROI_path_cal[ri][rj] = path_val
                ALL_ROI_path_cal[rj][ri] = -path_val


                # path_output_path = os.path.join(path_cal_save_folder, patient_id+'_'+str(ri)+'_'+str(rj)+'.txt')
                # save_mat_to_txt(path_cal, path_output_path)
                # 不存储，直接计算路径

    return retp,ALL_ROI_path_cal



def warp_sequences(seq_a, seq_b, path):
    '''
    seq_a, seq_b: 两个原始的序列
    path: 两个序列之间的规整路径
    path：(1,1),(1,2),(2,3)...(n,n)
    还原规整后的序列
    '''
    aligned_a = []
    aligned_b = []

    for (i, j) in path:
        aligned_a.append(seq_a[i])  # 减1是因为path是基于dtw_matrix的，它比seq_a和seq_b大1
        aligned_b.append(seq_b[j])
    return aligned_a, aligned_b


def pearson_correlation(seq1, seq2):
    """计算两个序列之间的皮尔森相关系数"""
    if(np.std(seq1)==0 or np.std(seq1)==0):
        print("存在标准差为0的序列！")
    return np.corrcoef(seq1, seq2)[0, 1]

def save_mat_to_txt(matrix, output_path):
    np.savetxt(output_path, matrix,delimiter=' ')

def process_one_file(filename,window_size,stride, in_folder, out_folder):
    # 输入输出文件路径的定义，矩阵的读入（已经转置）
    in_file_path = os.path.join(in_folder, filename)
    outp_file_path = os.path.join(out_folder, filename)
    matrix = load_one_file(in_file_path)

    path_cal_save_folder = out_folder +'path_cal/'
    if not os.path.exists(path_cal_save_folder):
        os.makedirs(path_cal_save_folder)

    patient_id = filename.replace('ROISignal_','')
    patient_id = patient_id.replace('_CN.txt', '')
    patient_id = patient_id.replace('_AD.txt', '')

    # 计算规整后的皮尔森相关系数矩阵，并存储
    retp,path_cal = cal_window_D(matrix, 90,window_size, stride,path_cal_save_folder,patient_id)
    save_mat_to_txt(retp, outp_file_path)
    path_output_path = os.path.join(out_folder, filename+'_path_cal.txt')
    save_mat_to_txt(path_cal, path_output_path)

def main():
    # 遍历文件夹中的所有文件
    for filename in os.listdir(in_folder):
        if filename.endswith(".txt"):
            for ws in window_size:
                for st in stride:
                    # 遍历窗口大小和步长
                    now_WDO_folder = os.path.join(out_folder, f"{ws}_{st}")
                    if not os.path.exists(now_WDO_folder):
                        os.makedirs(now_WDO_folder)
                    # 直接调用处理函数
                    process_one_file(filename, ws, st, in_folder, now_WDO_folder)
                    print(f"处理文件{filename}完成！")

if __name__ == '__main__':
    main()