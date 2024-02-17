"""
本文件是HDT工具包的模型训练模块。
"""
from scipy.stats import randint, uniform, expon, gamma
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from split import sample_partitioning_based_on_joint_x_y_distance as spxy
import pandas as pd
import numpy as np
import math
import random


def train_spectral_table(
        X, y,
        split_method="random",
        n_splits=100,
        test_size=0.3,
        n_iter=1000,
        model="PLSR",
        selected_index=None,
        rg=None,
        param_distributions=None,
        seed=None
):
    """
    该函数实现了对光谱表格数据进行多重随机划分训练模型的功能。

    输入：
    - X: 自变量的数据集。
    - y: 因变量的数据集。
    - split_method: 数据集划分方式，可选：spxy、random。
    - n_splits: 随机划分的次数。
    - test_size: 测试集划分比例。
    - n_iter: 每次训练模型时，从预设范围随机搜索参数的次数。
    - model: 字符串，所选择的模型。
    - selected_index: 列表，特征选择结果。
    - rg: 生成随机数种子的最大范围。
    - param_distributions: 自定义模型参数搜索范围。
    - seed: 随机数种子。

    输出：
    - result: 当随机划分次数为1时返回一个包括训练参数、模型参数、训练结果的字典。
    - df_results: 当随机划分次数大于1时返回一个包括训练参数、模型参数、训练结果的表格。
    """
    # 确定随机数种子生成范围
    if not rg:
        rg = 2 ** 32 - 1
    if type(selected_index) == type([]):
        X = X[:, selected_index]
        print(f"{X.shape}:{selected_index}")
    else:
        selected_index = range(X.shape[1])
    # 确定模型训练搜索参数范围
    model_param_distributions = {
        'PLSR': {
            'n_components': randint(1, X.shape[1]),
            'scale': [True, False],
            'max_iter': randint(50, 500),
            'tol': uniform(1e-5, 1e-2),
        },
        'KRR': {
            'alpha': uniform(loc=0, scale=1),
            'kernel': ['rbf'],
            'gamma': expon(scale=1),
        },
        'SVR': {
            'kernel': ['linear', 'rbf'],
            'C': expon(loc=0, scale=5),
            'gamma': ['scale', 'auto'] + list(gamma(0.1, 1).rvs(size=5)),
            'epsilon': uniform(loc=0, scale=0.5),
        },
        'KNN': {
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['manhattan', 'euclidean', 'minkowski']
        },
        'RF': {
            'n_estimators': randint(50, 100),
            'max_depth': randint(1, 20),
            'min_samples_leaf': randint(2, 10),
            'min_samples_split': randint(2, 10),
            'max_features': [1.0, 'sqrt', 'log2']
        },
        'BR': {
            'n_estimators': randint(50, 100),
            'max_features': np.arange(0.1, 1.1, 0.1)
        },
        'ABR': {
            'n_estimators': randint(10, 100),
            'learning_rate': uniform(0.01, 0.5),
        }
    }
    if not param_distributions:
        param_distributions = model_param_distributions[model]

    results = []
    if model != 'PLSR':
        y = y.ravel()
    n_splits = 1 if split_method == 'spxy' else n_splits

    for _ in tqdm(range(n_splits)):
        # 随机生成参数
        if not seed:
            random_state_split = random.randint(0, rg)
        else:
            random_state_split = seed
        random_state_search = random.randint(0, rg)

        # 划分训练集和测试集
        if split_method == 'spxy':
            train_index, test_index = spxy(X, y, test_size=test_size)
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            random_state_split = "spxy"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state_split
            )

        # 定义模型解释器
        model_estimator = {
            'PLSR': PLSRegression(),
            'KRR': KernelRidge(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'RF': RandomForestRegressor(),
            'BR': BaggingRegressor(),
            'ABR': AdaBoostRegressor()
        }
        estimator = model_estimator[model]

        # 定义随机搜索
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state_search,
            cv=5,
            n_jobs=8,
            scoring='neg_mean_squared_error'
        )

        # 拟合模型
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_

        # 在全体训练数据和测试数据上进行预测并计算指标
        y_train_pred = random_search.predict(X_train)
        y_test_pred = random_search.predict(X_test)
        train_r2_full = r2_score(y_train, y_train_pred)
        train_rmse_full = mean_squared_error(y_train, y_train_pred, squared=False)
        test_r2_full = r2_score(y_test, y_test_pred)
        test_rmse_full = mean_squared_error(y_test, y_test_pred, squared=False)
        train_SD = math.sqrt(sum((y_train - y_train_pred.mean()) ** 2 / (len(y_train) - 1)))
        test_SD = math.sqrt(sum((y_test - y_test_pred.mean()) ** 2 / (len(y_train) - 1)))
        train_rpd_full = train_SD / train_rmse_full
        test_rpd_full = test_SD / test_rmse_full

        # 记录模型参数和性能指标
        result = {
            'random_state_split': random_state_split,
            'random_state_search': random_state_search,
            'best_params': best_params,
            'train_r2_full': train_r2_full,
            'train_rmse_full': train_rmse_full,
            'test_r2_full': test_r2_full,
            'test_rmse_full': test_rmse_full,
            'train_rpd_full': train_rpd_full,
            'test_rpd_full': test_rpd_full,
            "feature_number": len(selected_index)
        }
        # 返回结果
        if n_splits == 1:
            return result
        results.append(result)
    df_results = pd.DataFrame(results)
    return df_results
