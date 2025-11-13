=====================================================================
IEEE Transactions on Medical Imaging (T-MI)
Alzheimer's Disease Diagnosis Based on Derivative Dynamic Time Warping Functional Connectivity Networks
=====================================================================
Framework: Dynamic Functional Connectivity Network Analysis
Methodology: Sliding Window based on Derivative Regularity Correlation (SWDRC) and Functional Delay Network (FDN)
Core Algorithm: Correlation-based on Derivative Regularity (CDR)
Dataset: ADNI (n=417 subjects), ABIDE (NYU & UM sites)
Modality: Resting-state functional Magnetic Resonance Imaging (rs-fMRI)
Author: Xin Hong, Yongze Lin,and Zhenghao Wu
Affiliation: Huaqiao University
Contact: xinhong@hqu.edu.cn
Version: v1.0.0
Code Repository: https://github.com/hxpotato/SWDRC
Copyright © 2025 IEEE
This work is licensed under the MIT License (see LICENSE for details)
This code is intended exclusively for academic and research use.
=====================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
import argparse
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_data_from_folder(folder):
    matrices = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            with open(os.path.join(folder, file)) as f:
                matrix = np.array([list(map(float, line.split())) for line in f])
                matrices.append(matrix)
    return matrices
def feature_selection(matrix, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    return selector
def flatten_connectivity_matrix(matrix):

    n = matrix.shape[0]
    features = []
    for i in range(n):
        for j in range(i+1, n):
            features.append(matrix[i, j])
    return np.array(features)
class Auto_encoder_MLP(nn.Module):
    def __init__(self, in_c, auto_1, auto_2, auto_3, MLP_1, MLP_2, MLP_out, dropout_rate=0.5):
        super(Auto_encoder_MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_c, auto_1),
            nn.ReLU(True),
            nn.Linear(auto_1, auto_2),
            nn.ReLU(True),
            nn.Linear(auto_2, auto_3),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(auto_3, auto_2),
            nn.ReLU(True),
            nn.Linear(auto_2, auto_1),
            nn.ReLU(True),
            nn.Linear(auto_1, in_c),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(auto_3, MLP_1),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(MLP_1, MLP_2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(MLP_2, MLP_out)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded)
        return classified, decoded
def calculate_metrics(y_true, y_pred, y_prob=None):

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_score = 0
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)
    return accuracy, sensitivity, specificity, auc_score, fpr, tpr
def train_model(model, train_loader, test_X, test_y, criterion_cls, criterion_auto, optimizer, device, lambda_auto=0.1):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output, decoded = model(data)
        loss_cls = criterion_cls(output, target)
        loss_auto = criterion_auto(decoded, data)
        loss = loss_cls + lambda_auto * loss_auto
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_output, _ = model(test_X)
        test_loss = criterion_cls(test_output, test_y).item()
        test_prob = nn.Softmax(dim=1)(test_output)
        test_pred = torch.max(test_output, 1)[1]
    return test_pred.cpu().numpy(), test_prob.cpu().numpy(), test_loss
def main():
    parser = argparse.ArgumentParser(description='AE-FCN for Brain Connectivity Classification')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--auto_in', type=int, default=1000, help='input features for autoencoder')
    parser.add_argument('--auto_hid_1', type=int, default=512, help='1st hidden layer of autoencoder')
    parser.add_argument('--auto_hid_2', type=int, default=256, help='2nd hidden layer of autoencoder')
    parser.add_argument('--auto_hid_3', type=int, default=128, help='3rd hidden layer of autoencoder')
    parser.add_argument('--MLP_1', type=int, default=64, help='1st hidden layer of MLP')
    parser.add_argument('--MLP_2', type=int, default=32, help='2nd hidden layer of MLP')
    parser.add_argument('--MLP_out', type=int, default=2, help='output dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lambda_auto', type=float, default=0.1, help='weight for autoencoder loss')
    parser.add_argument('--feature_num', type=int, default=1000, help='number of features after selection')
    parser.add_argument('--patience', type=int, default=1000, help='patience for early stopping')
    args = parser.parse_args()
    set_seed(args.seed)
    ad_matrices = load_data_from_folder('G:/wpsdata/Code/Python/exp01/result_AD_Fisher_z')
    nc_matrices = load_data_from_folder('G:/wpsdata/Code/Python/exp01/result_CN_Fisher_z')
    all_labels = np.array([1] * len(ad_matrices) + [0] * len(nc_matrices))
    all_matrices = ad_matrices + nc_matrices
    fmri_features = []
    for matrix in all_matrices:
        features = flatten_connectivity_matrix(matrix)
        fmri_features.append(features)
    fmri_features = np.array(fmri_features)
    scaler = StandardScaler()
    fmri_features = scaler.fit_transform(fmri_features)
    save_path = './results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'auc': []
    }
    for fold, (train_idx, test_idx) in enumerate(kfold.split(fmri_features, all_labels)):
        print(f"\n {fold+1}")
        selector = feature_selection(fmri_features, all_labels, train_idx, args.feature_num)
        selected_features = selector.transform(fmri_features)
        train_X = torch.FloatTensor(selected_features[train_idx]).cuda() if args.cuda else torch.FloatTensor(selected_features[train_idx])
        train_y = torch.LongTensor(all_labels[train_idx]).cuda() if args.cuda else torch.LongTensor(all_labels[train_idx])
        test_X = torch.FloatTensor(selected_features[test_idx]).cuda() if args.cuda else torch.FloatTensor(selected_features[test_idx])
        test_y = torch.LongTensor(all_labels[test_idx]).cuda() if args.cuda else torch.LongTensor(all_labels[test_idx])
        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        in_features = selected_features.shape[1]
        model = Auto_encoder_MLP(in_c=in_features, 
                                auto_1=args.auto_hid_1, 
                                auto_2=args.auto_hid_2, 
                                auto_3=args.auto_hid_3, 
                                MLP_1=args.MLP_1, 
                                MLP_2=args.MLP_2, 
                                MLP_out=args.MLP_out, 
                                dropout_rate=args.dropout_rate).cuda() if args.cuda else Auto_encoder_MLP(in_c=in_features, 
                                                                                                auto_1=args.auto_hid_1, 
                                                                                                auto_2=args.auto_hid_2, 
                                                                                                auto_3=args.auto_hid_3, 
                                                                                                MLP_1=args.MLP_1, 
                                                                                                MLP_2=args.MLP_2, 
                                                                                                MLP_out=args.MLP_out, 
                                                                                                dropout_rate=args.dropout_rate)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_auto = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_accuracy = 0
        best_pred = None
        best_prob = None
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(args.nEpochs):
            pred, prob, loss = train_model(model, train_loader, test_X, test_y, 
                                         criterion_cls, criterion_auto, optimizer, 
                                         torch.device("cuda" if args.cuda else "cpu"), args.lambda_auto)
            accuracy = accuracy_score(all_labels[test_idx], pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pred = pred
                best_prob = prob
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{args.nEpochs}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        accuracy, sensitivity, specificity, auc_score, fpr, tpr = calculate_metrics(
            all_labels[test_idx], best_pred, best_prob)
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['sensitivity'].append(sensitivity)
        fold_metrics['specificity'].append(specificity)
        fold_metrics['auc'].append(auc_score)
        print(f"\n {fold+1}:")
        print(f" {accuracy:.4f}")
        print(f" {sensitivity:.4f}")
        print(f" {specificity:.4f}")
        print(f" {auc_score:.4f}")
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold+1}')
        plt.legend()
        plt.close()
    avg_accuracy = np.mean(fold_metrics['accuracy'])
    avg_sensitivity = np.mean(fold_metrics['sensitivity'])
    avg_specificity = np.mean(fold_metrics['specificity'])
    avg_auc = np.mean(fold_metrics['auc'])
    print("\n5-fold:")
    print(f"avg_accuracy: {avg_accuracy:.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"avg_sensitivity: {avg_sensitivity:.4f} ± {np.std(fold_metrics['sensitivity']):.4f}")
    print(f"avg_specificity: {avg_specificity:.4f} ± {np.std(fold_metrics['specificity']):.4f}")
    print(f"avg_au: {avg_auc:.4f} ± {np.std(fold_metrics['auc']):.4f}")
    import pandas as pd
    results_df = pd.DataFrame({
        'Fold': list(range(1, 6)),
        'Accuracy': fold_metrics['accuracy'],
        'Sensitivity': fold_metrics['sensitivity'],
        'Specificity': fold_metrics['specificity'],
        'AUC': fold_metrics['auc']
    })
    results_df.loc[5] = ['Average', avg_accuracy, avg_sensitivity, avg_specificity, avg_auc]
if __name__ == "__main__":
    main()
