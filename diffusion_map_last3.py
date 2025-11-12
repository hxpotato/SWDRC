import os
import numpy as np
import itertools
from sklearn.covariance import GraphicalLasso
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.linalg


##############################################################################
#                              工具函数与缓存函数                            #
##############################################################################

def normalize_data(F):
    """
    标准化每一行数据, 使其具有零均值和单位方差
    参数:
      F: (p x m) 的 fMRI 矩阵
    返回:
      F_norm: (p x m) 标准化后的矩阵
    """
    return (F - np.mean(F, axis=1, keepdims=True)) / np.std(F, axis=1, keepdims=True)


def compute_sample_covariance(F_norm):
    """
    计算样本协方差矩阵 C = (1/m) * F_norm * F_norm^T
    参数:
      F_norm: (p x m)
    返回:
      C: (p x p) 的对称矩阵
    """
    m = F_norm.shape[1]
    return (F_norm @ F_norm.T) / m


def select_best_lambda(C, lambdas, n_samples):
    """
    使用 BIC 在给定 lambdas 中选最佳，返回逆协方差矩阵 S
    参数:
      C: (p x p) 的样本协方差矩阵
      lambdas: 备选的 lambda 值列表
      n_samples: 样本数量, 用于 BIC 计算
    返回:
      (S, best_lambda)
    """
    best_bic = np.inf
    best_S = None
    best_lam = None

    for lam in lambdas:
        try:
            model = GraphicalLasso(alpha=lam, max_iter=1000)
            model.fit(C)
            S = model.precision_
            log_likelihood = model.score(C)  # sklearn 的score返回的是对数似然的近似
            k = np.sum(S != 0)  # 非零元素个数
            bic = -2 * log_likelihood * n_samples + k * np.log(n_samples)
            if bic < best_bic:
                best_bic = bic
                best_S = S
                best_lam = lam
        except:
            continue

    return best_S, best_lam


# 计算Log-Euclidean距离
def log_euclidean_distance(R_i, R_j):
    """
    计算Log-Euclidean距离
    d_leu(R_i, R_j) = || log(R_i) - log(R_j) ||_F
    """
    log_R_i = scipy.linalg.logm(R_i)
    log_R_j = scipy.linalg.logm(R_j)
    diff = log_R_i - log_R_j
    distance = np.linalg.norm(diff, 'fro')
    return distance


# 计算Cholesky距离
def cholesky_distance(R_i, R_j):
    """
    计算Cholesky距离
    d_ck(R_i, R_j) = || L_i - L_j ||_F
    其中 L_i 和 L_j 是 R_i 和 R_j 的Cholesky分解的下三角部分
    """
    try:
        L_i = np.linalg.cholesky(R_i)
        L_j = np.linalg.cholesky(R_j)
        diff = L_i - L_j
        distance = np.linalg.norm(diff, 'fro')
    except np.linalg.LinAlgError:
        # 如果Cholesky分解失败（矩阵不是正定的），返回一个很大的距离
        distance = np.inf
    return distance


# 计算Euclidean距离
def euclidean_distance(R_i, R_j):
    """
    计算传统欧几里得距离
    d_eu(R_i, R_j) = || R_i - R_j ||_F
    """
    diff = R_i - R_j
    distance = np.linalg.norm(diff, 'fro')
    return distance


def diffusion_map(W, d):
    """
    执行 Diffusion Maps 降维
    参数:
      W: (n x n) 相似性矩阵
      d: 降维目标维度 (d << n)
    返回:
      (Y, K, phi, sorted_eigs):
        Y: (n x d), DM 的低维坐标
        K: (n x n), Markov 矩阵
        phi: (n, ), 平稳分布
        sorted_eigs: (vals, vecs), 特征值和特征向量(从大到小排序)
    """
    n = W.shape[0]
    row_sum = np.sum(W, axis=1)
    Q = np.diag(row_sum)
    inv_row_sum = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=(row_sum != 0))
    K = inv_row_sum[:, np.newaxis] * W
    total_sum = np.sum(row_sum)
    phi = row_sum / total_sum

    vals, vecs = np.linalg.eig(K)
    idx_sorted = np.argsort(-vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]

    # 跳过第0个(对应最大特征值 ~ 1)
    nontrivial_vals = vals_sorted[1:d + 1]
    nontrivial_vecs = vecs_sorted[:, 1:d + 1]
    Y = nontrivial_vecs * nontrivial_vals

    sorted_eigs = (vals_sorted, vecs_sorted)
    return Y, K, phi, sorted_eigs


##############################################################################
#                           缓存函数 1: 计算 / 读取 R_matrices                #
##############################################################################

def get_or_compute_R_matrices(
        data_list,
        lambda_candidates,
        cache_dir="cache",
        tag="default"
):
    """
    对每个被试计算 R 矩阵 (GraphicalLasso + invert), 若已存在相应缓存则直接加载。
    参数:
      data_list: list of (p x m) 矩阵
      lambda_candidates: 备选的 lambda 值列表
      cache_dir: 缓存文件夹
      tag: 附加标签, 用于区分不同数据集或不同设置

    返回:
      R_matrices: list of (p x p) SPD 矩阵
    """
    import hashlib
    import pickle

    # 构建一个简要的 hash, 用于标识本次设置: 数据规模 + lambda_candidates
    n = len(data_list)
    if n == 0:
        raise ValueError("data_list 为空！")

    p, m = data_list[0].shape
    lam_str = "_".join(map(str, lambda_candidates))

    # 简单拼接关键参数，用于生成文件名
    param_str = f"{tag}_n{n}_p{p}_m{m}_lam{lam_str}"
    # 生成 file_path
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"R_matrices_{param_str}.pkl")

    # 如果文件存在, 直接加载
    if os.path.exists(file_path):
        print(f"[INFO] 已检测到 R_matrices 缓存文件: {file_path}, 直接加载...")
        with open(file_path, "rb") as f:
            R_matrices = pickle.load(f)
        return R_matrices

    # 否则重新计算
    print(f"[INFO] 未检测到 R_matrices 文件 {file_path}, 需重新计算...")

    R_matrices = []
    for i, F in enumerate(data_list):
        F_norm = normalize_data(F)
        C = compute_sample_covariance(F_norm)
        S, best_lam = select_best_lambda(C, lambda_candidates, m)
        if S is None:
            print(f"  Subject {i} failed GraphicalLasso. Use identity fallback.")
            S = np.eye(p)
        try:
            R = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"  Subject {i} failed matrix invert. Use identity fallback.")
            R = np.eye(p)
        R_matrices.append(R)

    # 将 R_matrices 保存到文件
    with open(file_path, "wb") as f:
        pickle.dump(R_matrices, f)
    print(f"[INFO] R_matrices 已保存到: {file_path}")

    return R_matrices


##############################################################################
#                      缓存函数 2: 计算 / 读取 “距离数组 + sigma”              #
##############################################################################

def get_or_compute_distances_and_sigma(
        R_matrices,
        distance_func,
        distance_name="logeucl",
        cache_dir="cache",
        tag="default"
):
    """
    基于 R_matrices 和给定的距离函数, 计算所有 pairwise 的距离, 再估计 sigma (如中位数^2)。
    若已存在缓存文件则直接加载; 否则重新计算并保存。

    返回:
      distances: list of 所有 pairwise 距离
      sigma: float, 用距离中位数的平方 (或可自行修改)
    """
    import pickle

    n = len(R_matrices)
    if n <= 1:
        raise ValueError("R_matrices 数量不足, 无法计算距离或 sigma.")

    # 构造文件名, 不包含 sigma, 因为 sigma 尚未计算出来
    param_str = f"{tag}_n{n}_{distance_name}"
    file_path = os.path.join(cache_dir, f"distances_{param_str}.pkl")

    if os.path.exists(file_path):
        print(f"[INFO] 已检测到 distances 缓存文件: {file_path}, 直接加载...")
        with open(file_path, "rb") as f:
            saved = pickle.load(f)
        distances = saved["distances"]
        sigma = saved["sigma"]
        return distances, sigma

    print(f"[INFO] 未检测到 distances 文件 {file_path}, 需重新计算...")

    distances = []
    for i, j in itertools.combinations(range(n), 2):
        d_ij = distance_func(R_matrices[i], R_matrices[j])
        if not np.isinf(d_ij):
            distances.append(d_ij)

    if len(distances) == 0:
        print("[WARN] 全部距离为inf，强制 sigma=1.0")
        sigma = 1.0
    else:
        median_ = np.median(distances)
        sigma = median_ ** 2

    # 缓存到文件
    os.makedirs(cache_dir, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump({"distances": distances, "sigma": sigma}, f)
    print(f"[INFO] distances + sigma 已保存到: {file_path}")

    return distances, sigma


##############################################################################
#                   缓存函数 3: 计算 / 读取 W 矩阵 (含 sigma)                #
##############################################################################

def compute_similarity_matrix_from_distances(distances, n, distance_func, R_matrices, sigma):
    """
    从已知的 pairwise distances 列表/及 sigma，构造 (n x n) 相似性矩阵 W。
    因为 distances 是 pairwise 形式，需要对应 R_matrices 的顺序。
    我们可以选择再次计算（节省内存）: 直接用 distance_func 动态计算 d(i,j) 再映射到 exp(-d^2/sigma)。
    或者自己维护一个 pairwise distance 矩阵 D，然后再映射到 W。
    此处演示“再次计算”的流程——通常不会损失太多时间，因为已经有 sigma, 距离函数也有, R_matrices 也有。
    """

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            d_ij = distance_func(R_matrices[i], R_matrices[j])
            if np.isinf(d_ij):
                val = 0.0
            else:
                val = np.exp(- (d_ij ** 2) / sigma)
            W[i, j] = val
            W[j, i] = val
    return W


def get_or_compute_W(
        R_matrices,
        distance_func,
        sigma,
        distance_name="logeucl",
        cache_dir="cache",
        tag="default"
):
    """
    基于 R 矩阵列表、给定距离度量名称 & sigma, 计算或加载 W 矩阵。
    文件名中直接包含 sigma, 以满足需求:
      "这个参数直接在W矩阵的文件名中保存、体现"

    返回:
      W: (n x n) 相似性矩阵
    """
    import pickle

    n = len(R_matrices)
    sigma_str = f"{sigma:.4f}"  # 保留4位小数
    param_str = f"{tag}_n{n}_{distance_name}_sigma_{sigma_str}"
    file_path = os.path.join(cache_dir, f"W_matrix_{param_str}.npy")

    if os.path.exists(file_path):
        print(f"[INFO] 已检测到 W 矩阵文件: {file_path}, 直接加载...")
        W = np.load(file_path)
        return W

    print(f"[INFO] 未检测到 W 文件 {file_path}, 需重新计算...")
    W = compute_similarity_matrix_from_distances(
        distances=None,  # 这里可不传, 我们直接用 distance_func 重算
        n=n,
        distance_func=distance_func,
        R_matrices=R_matrices,
        sigma=sigma
    )
    np.save(file_path, W)
    print(f"[INFO] 已保存 W 矩阵到: {file_path}")

    return W


##############################################################################
#                          主实验流程: 演示自动缓存                           #
##############################################################################

def generate_dummy_data(n, p, m, seed=42):
    """
    生成 n 个 (p x m) 的随机矩阵, 仅作示例。真实场景请换成你的 fMRI 数据加载。
    """
    np.random.seed(seed)
    data_list = []
    for _ in range(n):
        # 随机生成 (p x m) 矩阵
        X = np.random.randn(p, m)
        data_list.append(X)
    return data_list

import os
def load_mats(folder, patient_type):
    # 加载某类别的特征矩阵
    class_matrices = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt') and patient_type in file_name:
            file_path = os.path.join(folder, file_name)
            z_matrix = np.loadtxt(file_path)
            # 翻转矩阵，使得时间点在第一维度
            z_matrix = z_matrix.T
            # 限制时间点数量 (这里示例取前 135 个时间点)
            # z_matrix = z_matrix[:, :135]
            # 限制 ROI 数量 (这里示例取前 90 个 ROI)
            z_matrix = z_matrix[:90, :]
            class_matrices.append(z_matrix)
    return class_matrices
# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}

def main():
    ad_roi_matrices = load_mats('G:/WPSData/Code/Python/exp01/ROIsingal', 'AD')
    cn_roi_matrices = load_mats('G:/WPSData/Code/Python/exp01/ROIsingal', 'CN')
    X = np.array(cn_roi_matrices + ad_roi_matrices)
    y = np.array([0] * len(cn_roi_matrices) + [1] * len(ad_roi_matrices))
    n, p, m = 83, 90, 135
    data_list = X
    labels = y

    # ========== 2) 计算 / 读取 R_matrices ==========
    lambda_candidates = [0.001,0.01, 0.1, 1.0, 10.0]
    R_matrices = get_or_compute_R_matrices(
        data_list=data_list,
        lambda_candidates=lambda_candidates,
        cache_dir="cache",
        tag="Exp1"
    )

    # ========== 3) 计算 / 读取 “距离数组 + sigma” ==========
    distance_func = euclidean_distance
    # euclidean_distance, log_euclidean_distance, cholesky_distance
    distance_name = "cholesky_distance"  # 用于文件命名
    distances, sigma = get_or_compute_distances_and_sigma(
        R_matrices=R_matrices,
        distance_func=cholesky_distance,
        distance_name=distance_name,
        cache_dir="cache",
        tag="Exp2"
    )
    print(f"[INFO] 本次估计的 sigma = {sigma:.4f}")

    # ========== 4) 计算 / 读取 W ==========
    W = get_or_compute_W(
        R_matrices=R_matrices,
        distance_func=distance_func,
        sigma=sigma,
        distance_name=distance_name,
        cache_dir="cache",
        tag="Exp2"
    )
    
    d_dm = 30
    Y, K, phi, sorted_eigs = diffusion_map(W, d_dm)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = SVC(kernel='rbf')
    scores = []
    for tr_idx, te_idx in kf.split(Y, labels):
        X_tr, X_te = Y[tr_idx], Y[te_idx]
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        scores.append(acc)
    print(f"[RESULT] DM+SVM (5-fold CV) Acc = {np.mean(scores):.4f}")

    from sklearn.model_selection import GridSearchCV
    # 定义 SVM 分类器
    svm = SVC()

    # 定义 GridSearchCV 对象
    grid_search = GridSearchCV(svm, param_grid, cv=kf, scoring='accuracy')

    # 进行超参数搜索
    grid_search.fit(Y, labels)

    # 输出最佳参数和对应的准确率
    print("最佳参数: ", grid_search.best_params_)
    print("最佳交叉验证准确率: {:.2f}%".format(grid_search.best_score_ * 100))

if __name__ == "__main__":
    main()
