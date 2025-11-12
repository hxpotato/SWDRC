import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
def vectorize_fcn(X):
    n_samples, n_roi, _ = X.shape
    triu_indices = np.triu_indices(n_roi, k=1)
    X_vect = X[:, triu_indices[0], triu_indices[1]]  
    return X_vect
def main():
    X = np.load('significant_matrices.npy')
    y = np.array([1] * 33 + [0] * 50)
    X_vect = vectorize_fcn(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vect)
    from pydiffmap import diffusion_map as dm
    diffusion = dm.DiffusionMap.from_sklearn(n_evecs=10, epsilon='bgh')
    X_dm = diffusion.fit_transform(X_scaled)
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    aucs = []
    sensitivities = []
    specificities = []
    fold = 1
    for train_idx, test_idx in cv.split(X_dm, y):
        print(f"执行第 {fold} 折交叉验证...")
        X_train, X_test = X_dm[train_idx], X_dm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        svc = SVC(probability=True, random_state=42)
        grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)  
        specificity = tn / (tn + fp)  
        accuracies.append(acc)
        aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        fold += 1
    print("\n五折交叉验证性能指标汇总:")
    print(f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"平均AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"平均敏感性: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"平均特异性: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")
if __name__ == '__main__':
    main()
