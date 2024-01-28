import os
import random
import shutil
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import torch
from matplotlib import pyplot as plt
from torcheval.metrics.functional import r2_score, mean_squared_error
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['courier new']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


def getFileName(path):
    ''' 获取指定目录下的所有指定后缀的文件名 '''

    fileNameList = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.hdr':
            print(i)
            fileNameList.append(i)

    return fileNameList


def match_then_insert(filename, match, content):
    """匹配后在该行追加
    :param filename: 要操作的文件
    :param match: 匹配内容
    :param content: 追加内容
    """
    with open(filename, mode='rb+') as f:
        while True:
            try:
                line = f.readline()  # 逐行读取
            except IndexError:  # 超出范围则退出
                break
            line_str = line.decode().splitlines()[0]
            if line_str == match:
                f.seek(-len(line), 1)  # 光标移动到上一行
                rest = f.read()  # 读取余下内容
                f.seek(-len(rest), 1)  # 光标移动回原位置
                f.truncate()  # 删除余下内容
                content = content + '\n'
                f.write(content.encode())  # 插入指定内容
                f.write(rest)  # 还原余下内容
                break


def raw2envi(rootPath):
    fileNameList = getFileName(rootPath)

    for idx, fileName in enumerate(fileNameList):
        print(idx)
        print(fileName)
        filePath = os.path.join(rootPath, fileName)
        match_then_insert(filePath, match='wavelength = {', content='byte order = 0')


def PCA_reshape(data, n_components):
    """
    使用PCA降维
    """
    # 将数据重构为二维数组，以进行PCA
    nsamples, nx, ny = data.shape
    d2_data = data.reshape((nsamples * nx, ny))

    # 标准化数据
    scaler = StandardScaler()
    d2_data_std = scaler.fit_transform(d2_data)

    # 应用PCA
    # 假设我们想将数据维度降至20
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(d2_data_std)

    # 如果需要，可以将降维后的数据重塑回接近原始数据的三维形状
    _, my = principalComponents.shape
    # 例如，重塑为 100x100x20 的数组
    final_data = principalComponents.reshape((nsamples, nx, my))
    return final_data


def crop_data(BW, data, crop_save=False):
    """
    裁剪数据
    """
    binary_image = BW.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_contours = sorted(contours, key=len, reverse=True)[0].squeeze()
    # 获取最大外接矩形
    x1 = max_contours[:, 0].min()
    y1 = max_contours[:, 1].min()
    x2 = max_contours[:, 0].max()
    y2 = max_contours[:, 1].max()
    # 裁剪数据
    cropped_images = np.empty([y2 - y1, x2 - x1, data.shape[2]])
    for j in range(data.shape[2]):
        cropped_images[:, :, j:j + 1] = data[y1:y2, x1:x2, j:j + 1]
    # 保存数据
    if crop_save:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(max_contours[:, 0], max_contours[:, 1], s=1)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor='red', linestyle='--')
        ax.add_patch(rect)
        plt.imshow(binary_image)
        plt.xticks([])
        plt.yticks([])
        plt.close()
        return cropped_images, fig
    return cropped_images


def binary_data(read_data_src, k=1.0, crop_save=False, shape_save=False):
    """
    二值化数据
    """
    # 数据读取的文件夹
    rootPath = os.path.split(read_data_src)[0]
    save_bin_src = f'{rootPath}/binData'
    save_crop_src = f'{rootPath}/cropData'

    # 创建文件夹
    if not os.path.exists(save_bin_src):
        os.mkdir(save_bin_src)
    if not os.path.exists(save_crop_src):
        os.mkdir(save_crop_src)

    # 设置路径
    read_data_dir = os.listdir(read_data_src)
    read_data_hdr_name = [f"{read_data_src}/{hdr}" for hdr in read_data_dir if hdr.endswith('.hdr')]

    # 记录形状
    shape_list = []

    # 遍历提取
    for hdr in tqdm(read_data_hdr_name):
        filePath, fileName = os.path.split(hdr)
        raw = os.path.join(filePath, fileName.replace('hdr', 'raw'))
        bin = os.path.join(rootPath, 'binData', fileName.replace('hdr', 'png'))
        crop = os.path.join(rootPath, 'cropData', fileName.replace('hdr', 'npy'))

        # 读取光谱数据
        img = envi.open(hdr, raw)
        # 阈值分割
        BW = img[:, :, 111]
        mask = np.where(np.squeeze(BW) > k * BW.mean(), 1, 0)
        plt.imsave(bin, mask)

        # 裁剪数据
        if crop_save:
            cropped_images, fig = crop_data(mask, img, crop_save)
            png = bin.replace('.png', '-crop.png')
            np.save(crop, cropped_images)
            fig.savefig(png, bbox_inches='tight')
        else:
            cropped_images = crop_data(mask, img)
            np.save(crop, cropped_images)
        shape_list.append({
            "num": os.path.split(hdr)[-1].split('.')[0],
            "shape": cropped_images.shape
        })
    if shape_save:
        pd.DataFrame(shape_list).to_csv(f"{rootPath}/shape.csv", index=False)


def plot_classification_curve(train_curve, valid_curve, test_accuracy, xlabel="", ylabel="", title="", savepath=""):
    """
    绘制训练曲线
    """
    plt.figure(dpi=200)
    plt.plot(range(1, len(train_curve) + 1), train_curve, 'o-', label='train')
    plt.plot(range(1, len(valid_curve) + 1), valid_curve, 'o-', label='valid')
    plt.axhline(y=test_accuracy, xmin=0, xmax=1, label='test', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)


def plot_regression_curve(train_log, xlabel="", ylabel="", title="", savepath=""):
    train_rmse, valid_rmse, test_rmse = train_log['train_rmse'], train_log['valid_rmse'], \
        train_log['test_rmse'].tolist()[-1]
    train_r2, valid_r2, test_r2 = train_log['train_r2'], train_log['valid_r2'], train_log['test_r2'].tolist()[-1]
    train_rpd, valid_rpd, test_rpd = train_log['train_rpd'], train_log['valid_rpd'], train_log['test_rpd'].tolist()[-1]

    plt.figure(dpi=300)
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, 'o-', label='train_rmse', color='red')
    plt.plot(range(1, len(valid_rmse) + 1), valid_rmse, 'o-.', label='valid_rmse', color='red')
    plt.axhline(y=test_rmse, xmin=0, xmax=1, label='test_rmse', linestyle='--', color='red')

    plt.plot(range(1, len(train_r2) + 1), train_r2, 'o-', label='train_r2', color='blue')
    plt.plot(range(1, len(valid_r2) + 1), valid_r2, 'o-.', label='valid_r2', color='blue')
    plt.axhline(y=test_r2, xmin=0, xmax=1, label='test_r2', linestyle='--', color='blue')

    plt.plot(range(1, len(train_rpd) + 1), train_rpd, 'o-', label='train_rpd', color='green')
    plt.plot(range(1, len(valid_rpd) + 1), valid_rpd, 'o-.', label='valid_rpd', color='green')
    plt.axhline(y=test_rpd, xmin=0, xmax=1, label='test_rpd', linestyle='--', color='green')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)


def make_new_directory(directory_path):
    """
    检查目录是否存在，若存在则清空，若不存在则创建
    """
    if os.path.exists(directory_path):
        # 遍历目录中的文件和子文件夹，然后删除
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    # 如果是文件，删除文件
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 如果是文件夹，递归删除文件夹及其内容
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}: {e}")

    else:
        os.makedirs(directory_path)


def make_directory(directory_path):
    """
    检查目录是否存在，若不存在则创建
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def eval_classification_model(net, data_loader):
    """
    评估模型
    """
    correct = torch.tensor(0.)
    total = 0.
    # 将网络模式设置为评估
    net.eval()
    # 不计算梯度
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        loop.set_description(f'Eval Model')
        for i, data in loop:
            # 获取数据
            img, label = data
            # img, label = Variable(img), Variable(label)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
                # 预测标签
            out = net(img)
            # 获取预测的样本的标签
            _, predicted = torch.max(out.data, 1)
            # 计算样本总个数
            total += label.size(0)
            # 正确的个数
            correct += sum(predicted == label)
        # 这是验证集的准确率
        accuracy = correct.item() / total
        return net, accuracy


def eval_regression_model(net, data_loader):
    """
    评估回归类模型
    """
    predictedList = torch.Tensor()
    targetList = torch.Tensor()
    if torch.cuda.is_available():
        predictedList = predictedList.cuda()
        targetList = targetList.cuda()
    # 将网络模式设置为评估
    net.eval()
    # 不计算梯度
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        loop.set_description(f'Eval Model')
        for i, data in loop:
            # 获取数据
            img, target = data
            img, target = img.float(), target.float()
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
                # 预测标签
            out = net(img)
            predicted = out.data
            # 添加预测结果
            predictedList = torch.cat([predictedList, predicted])
            targetList = torch.cat([targetList, target])
        # 直接将predictedList的纬度转为targetList的纬度
        predictedList = predictedList.reshape(targetList.shape)
        # 添加完成，计算预测结果
        rmse = torch.sqrt(mean_squared_error(predictedList, targetList)).item()
        r2 = r2_score(predictedList, targetList).item()
        rpd = relative_percent_difference(predictedList, targetList).item()
        return net, rmse, r2, rpd


def split_classification_dataset(rootPath, seed=0, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    根据随机数种子，按一定比例随机划分分类任务的训练集、验证集和测试集
    """

    # 设置随机数种子
    random.seed(seed)

    # 训练集、验证集、测试集比例
    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= sum_ratio
    val_ratio /= sum_ratio
    test_ratio /= sum_ratio
    print(f'{train_ratio}:{val_ratio}:{test_ratio}')

    # 原始数据路径
    rawData = f'{rootPath}/rawData'
    # 划分后数据路径
    splitData = f'{rootPath}/splitData'

    # 检查文件夹
    make_new_directory(f'{splitData}/train')
    make_new_directory(f'{splitData}/valid')
    make_new_directory(f'{splitData}/test')

    # 遍历每一个文件夹
    for sub_dir in tqdm(os.listdir(rawData)):

        # 获取当前文件夹下的所有文件名
        sub_listdir = os.listdir(f"{rawData}/{sub_dir}")

        # 计算各部分的样本数量
        total_samples = len(sub_listdir)
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        test_samples = total_samples - train_samples - val_samples

        # 随机打乱
        random.shuffle(sub_listdir)

        # 获取训练集、验证集、测试集
        train_set = sub_listdir[:train_samples]
        val_set = sub_listdir[train_samples:train_samples + val_samples]
        test_set = sub_listdir[train_samples + val_samples:]

        # 确保划分比例正确
        assert len(train_set) == train_samples
        assert len(val_set) == val_samples
        assert len(test_set) == test_samples

        # 保存到目标文件夹
        for set, split_dir in zip([train_set, val_set, test_set], ['train', 'valid', 'test']):
            # 检查文件夹
            make_new_directory(f'{splitData}/{split_dir}/{sub_dir}')
            for img in set:
                # 保存文件
                shutil.copy2(f'{rawData}/{sub_dir}/{img}', f'{splitData}/{split_dir}/{sub_dir}/{img}')


def split_regression_dataset(csvPath, seed=0, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    根据随机数种子，按一定比例随机划分回归任务的训练集、验证集和测试集
    """
    # 设置随机数种子
    random.seed(seed)

    # 训练集、验证集、测试集比例
    sum_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= sum_ratio
    val_ratio /= sum_ratio
    test_ratio /= sum_ratio

    df_csv = pd.read_csv(csvPath)

    # 计算各部分的样本数量
    total_samples = len(df_csv)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples

    df_index = df_csv.index.tolist()
    random.shuffle(df_index)

    # 获取训练集、验证集、测试集
    train_set = df_index[:train_samples]
    val_set = df_index[train_samples:train_samples + val_samples]
    test_set = df_index[train_samples + val_samples:]

    # 确保划分比例正确
    assert len(train_set) == train_samples
    assert len(val_set) == val_samples
    assert len(test_set) == test_samples

    rootPath = os.path.split(csvPath)[0]
    df_csv.loc[train_set].to_csv(os.path.join(rootPath, 'train.csv'))
    df_csv.loc[val_set].to_csv(os.path.join(rootPath, 'valid.csv'))
    df_csv.loc[test_set].to_csv(os.path.join(rootPath, 'test.csv'))


def relative_percent_difference(predicted, target):
    """计算rpd"""
    return torch.sqrt(sum(target - predicted.mean()) ** 2 / (len(target) - 1))


def padding_array(original_array, target_shape):
    # 计算在每个维度上的填充量
    target_height, target_width, target_depth = target_shape

    pad_height = target_height - original_array.shape[0]
    pad_width = target_width - original_array.shape[1]
    pad_depth = target_depth - original_array.shape[2]

    # 使用np.pad在周围填充0
    padded_array = np.pad(original_array, ((0, pad_height), (0, pad_width), (0, pad_depth)), mode='constant')

    return padded_array


def preprocess(refPath):
    """
    二值化、裁剪、形状调整、主成分分析
    """
    rootPath = os.path.split(refPath)[0]
    shapePath = f'{rootPath}/shape.csv'
    cropPath = f'{rootPath}/cropData'
    pcaPath = f'{rootPath}/pcaData'

    # 二值化并裁剪数据
    binary_data(refPath, crop_save=True, shape_save=True)
    # 主成分分析
    PCA_dataset(cropPath, n_components=0.9)
    # 调整矩阵形状
    resize_array(pcaPath, shapePath)


def resize_array(rawPath, shapePath):
    resizePath = fr'{os.path.split(rawPath)[0]}/resizeData'

    if not os.path.exists(resizePath):
        os.mkdir(resizePath)

    shape = pd.read_csv(shapePath)["shape"]
    target_shape = np.array([eval(x) for x in shape.tolist()]).max(axis=0)

    for npy in os.listdir(rawPath):
        original_array = np.load(f"{rawPath}/{npy}")
        padded_array = padding_array(original_array, target_shape)
        np.save(f'{resizePath}/{npy}', padded_array)


def getFileName(path):
    ''' 获取指定目录下的所有指定后缀的文件名 '''

    fileNameList = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.hdr':
            print(i)
            fileNameList.append(i)

    return fileNameList


def match_then_insert(filename, match, content):
    """匹配后在该行追加
    :param filename: 要操作的文件
    :param match: 匹配内容
    :param content: 追加内容
    """
    with open(filename, mode='rb+') as f:
        while True:
            try:
                line = f.readline()  # 逐行读取
            except IndexError:  # 超出范围则退出
                break
            line_str = line.decode().splitlines()[0]
            if line_str == match:
                f.seek(-len(line), 1)  # 光标移动到上一行
                rest = f.read()  # 读取余下内容
                f.seek(-len(rest), 1)  # 光标移动回原位置
                f.truncate()  # 删除余下内容
                content = content + '\n'
                f.write(content.encode())  # 插入指定内容
                f.write(rest)  # 还原余下内容
                break


def raw2envi(rootPath):
    """标准envi化"""
    fileNameList = getFileName(rootPath)

    for idx, fileName in enumerate(fileNameList):
        print(idx)
        print(fileName)
        filePath = os.path.join(rootPath, fileName)
        match_then_insert(filePath, match='wavelength = {', content='byte order = 0')


def PCA_dataset(cropPath, n_components=0.9):
    """
    对裁剪后的数据进行主成分分析
    """
    rootPath = os.path.split(cropPath)[0]
    pcaPath = f'{rootPath}/pcaData'

    if not os.path.exists(pcaPath):
        os.mkdir(pcaPath)

    for file in os.listdir(cropPath):
        data = np.load(f'{cropPath}/{file}')
        dataPca = PCA_reshape(data, n_components)
        np.save(f'{pcaPath}/{file}', dataPca)
