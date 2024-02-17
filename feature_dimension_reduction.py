"""
本文件是HDT工具包的特征降维模块，包括竞争自适应重加权采样算法、连续投影法、主成分分析法。
"""
from sklearn.model_selection import KFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy
from split import sample_partitioning_based_on_joint_x_y_distance
from scipy.linalg import qr
import copy
import warnings

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def competitive_adapative_reweighted_sampling(X, y, N=50, f=20, cv=10):
    """
    该函数实现了基于竞争自适应重加权采样算法的特征选择功能。

    输入：
    - X: 自变量的数据集。
    - y: 因变量的数据集。
    - N: 迭代次数。
    - f: 最大主成分数目。
    - cv: 交叉验证折数。

    输出：
    - OptWave: 所选择的特征索引列表。
    """

    def pc_cross_validation(X, y, pc, cv):
        """
        该函数实现了基于交叉验证的主成分回归模型评估和最优特征数目查找功能。

        输入：
        - x: 自变量的数据集。
        - y: 因变量的数据集。
        - pc: 最大主成分数目。
        - cv: 交叉验证折数。

        输出：
        - RMSECV: 各主成分数目模型预测均方误差。
        - rindex: 最优主成分数目。
        """
        kf = KFold(n_splits=cv)
        RMSECV = []
        for i in range(pc):
            RMSE = []
            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                pls = PLSRegression(n_components=i + 1)
                pls.fit(x_train, y_train)
                y_predict = pls.predict(x_test)
                RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            RMSE_mean = np.mean(RMSE)
            RMSECV.append(RMSE_mean)
        rindex = np.argmin(RMSECV)
        return RMSECV, rindex

    def cross_validation(X, y, pc, cv):
        """
        该函数实现了基于交叉验证的给定特征数目的主成分模型评估功能。

        输入：
        - x: 自变量的数据集。
        - y: 因变量的数据集。
        - pc: 主成分数目。
        - cv: 交叉验证折数。

        输出：
        - RMSE_mean: 模型预测均方误差。
        - rindex: 最优主成分数目。
        """
        kf = KFold(n_splits=cv)
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=pc)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        return RMSE_mean

    p = 0.8
    m, n = X.shape
    u = np.power((n / 2), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 2)
    cal_num = np.round(m * p)
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    WaveNum = []
    RMSECV = []
    r = []
    for i in range(1, N + 1):
        r.append(u * np.exp(-1 * k * i))
        wave_num = int(np.round(r[i - 1] * n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData):
            WaveData = np.vstack((WaveData, d.reshape(1, -1)))
        else:
            WaveData = d

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        rmsecv, rindex = pc_cross_validation(xcal, ycal, f, cv)
        RMSECV.append(cross_validation(xcal, ycal, rindex + 1, cv))

    WAVE = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        WD = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
            else:
                WD[j] = wd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    return OptWave


def successive_projections_algorithm(X, y, test_size=0.3, split_method='random', m_min=1, m_max=None):
    '''
    该函数实现了基于连续投影法的特征选择功能。

    输入：
    - X: 自变量的数据集。
    - y: 因变量的数据集。
    - test_size: 测试集划分比例。
    - split_method: 数据集划分方法。
    - m_min: 特征选择最小值。
    - m_max: 特征选择最大值。

    输出：
    - var_sel: 所选择的特征索引列表。

    '''

    def variable_selection_validation(Xcal, ycal, Xval, yval, var_sel):
        '''
        该函数实现了验证连续投影法选择结果性能的功能。

        输入：
        - Xcal: 输入训练集。
        - ycal: 输出训练集。
        - Xval: 输入验证集。
        - yval: 输出验证集。
        - var_sel: 选择的变量。

        输出：
        - e: 误差。

        '''
        N, NV = Xcal.shape[0], Xval.shape[0]
        Xcal_ones = np.hstack([np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])
        b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
        X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
        yhat = X.dot(b)
        e = yval - yhat
        return e

    def variable_selection_qr(X, k, M):
        '''
        该函数基于QR分解实现了根据给定预测变量矩阵X找到最相关的M个变量的功能。

        输入：
        - X: 预测变量矩阵。
        - K: 投影操作的初始列的索引。
        - M: 结果包含的变量个数。

        输出：
        - SELk: 由投影操作生成的变量集的索引。
        '''

        X_projected = X.copy()
        norms = np.sum((X ** 2), axis=0)
        norm_max = np.amax(norms)
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]
        _, __, order = qr(X_projected, 0, pivoting=True)
        SELk = order[:M].T
        return SELk

    if split_method == 'spxy':
        train_index, test_index = sample_partitioning_based_on_joint_x_y_distance(X, y, test_size=test_size)
        Xcal, Xval = X[train_index], X[test_index]
        ycal, yval = y[train_index], y[test_index]
    else:
        Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=test_size)

    N, K = Xcal.shape

    if not m_max:
        m_max = min(N - 2, K)

    normalization_factor = np.std(Xcal, ddof=1, axis=0).reshape(1, -1)[0]

    Xcaln = np.empty((N, K))
    for k in range(K):
        x = Xcal[:, k]
        Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

    SEL = np.zeros((m_max, K))

    for k in range(K):
        SEL[:, k] = variable_selection_qr(Xcaln, k, m_max)

    PRESS = float('inf') * np.ones((m_max + 1, K))

    for k in range(K):
        for m in range(m_min, m_max + 1):
            var_sel = SEL[:m, k].astype(int)
            e = variable_selection_validation(Xcal, ycal, Xval, yval, var_sel)
            PRESS[m, k] = np.conj(e).T.dot(e)

    PRESSmin = np.min(PRESS, axis=0)
    m_sel = np.argmin(PRESS, axis=0)
    k_sel = np.argmin(PRESSmin)

    var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(int)

    Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
    b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
    std_deviation = np.std(Xcal2, ddof=1, axis=0)

    relev = np.abs(b * std_deviation.T)[1:]

    index_increasing_relev = np.argsort(relev, axis=0)
    index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

    PRESS_scree = np.empty(len(var_sel_phase2))
    e = None
    for i in range(len(var_sel_phase2)):
        var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
        e = variable_selection_validation(Xcal, ycal, Xval, yval, var_sel)

        PRESS_scree[i] = np.conj(e).T.dot(e)

    PRESS_scree_min = np.min(PRESS_scree)
    alpha = 0.25
    dof = len(e)
    fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
    PRESS_crit = PRESS_scree_min * fcrit

    i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
    i_crit = max(m_min, i_crit)
    var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

    return var_sel
