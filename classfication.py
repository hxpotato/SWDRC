import os
import numpy as np
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
def load_mats(folder,patient_type,delimiter):
    class_matrices=[]
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt') and patient_type in file_name:
            file_path = os.path.join(folder, file_name)
            z_matrix = np.loadtxt(file_path,delimiter=delimiter)
            class_matrices.append(z_matrix)
    class_matrices = np.array(class_matrices)
    return class_matrices
def extract_significant_features(p_values, threshold):
    significant_features = np.where(p_values < threshold)
    return significant_features
def classify_5z(features, labels):
    clf = SVC()
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, features, labels, cv=skf)
    print(f"五折准确率: {scores.mean() * 100}%")
    return scores.mean()
def classify_loo(features, labels):
    clf = SVC()
    loo = LeaveOneOut()
    scores = cross_val_score(clf, features, labels, cv=loo)
    print(f"留一法准确率: {scores.mean() * 100}%")
    return scores.mean()
def main():
    data_folder = './data/'
    p_val_folder = './pval/'
    threshold = [0.05, 0.01, 0.005, 0.001]
    folders = os.listdir(data_folder)
    csv_file = open('results.csv', 'w')
    csv_file.write("Name,P阈值,特征数量,五折,留一法\n")
    for folder in folders:
        name = folder
        print(f"Folder: {name}")
        infolder = os.path.join(data_folder, folder)
        CN_mats = load_mats(infolder, 'CN', delimiter=' ')
        AD_mats = load_mats(infolder, 'AD', delimiter=' ')
        p_val_matrix = np.loadtxt(os.path.join(p_val_folder, f'{name}.txt'), delimiter=' ')
        for t in threshold:
            print(f"Threshold: {t}")
            significant_features = extract_significant_features(p_val_matrix, t)
            feature_count = len(significant_features[0])
            if feature_count == 0:
                print("没有显著特征，跳过")
                continue
            important_data_1 = np.array(CN_mats)[:, significant_features[0], significant_features[1]]
            important_data_2 = np.array(AD_mats)[:, significant_features[0], significant_features[1]]
            important_data = np.concatenate((important_data_1, important_data_2), axis=0)
            labels = np.concatenate((np.zeros(CN_mats.shape[0]), np.ones(AD_mats.shape[0])))
            score_5z =classify_5z(important_data, labels)
            score_loo = classify_loo(important_data, labels)
            csv_file.write(f"{name},{t},{feature_count},{score_5z},{score_loo}\n")
if __name__ == '__main__':
    main()
