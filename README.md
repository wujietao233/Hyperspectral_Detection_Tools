# 使用说明

本工具包是基于Python实现的高光谱检测工具包，实现了高光谱图像数据ROI区域提取、高光谱数据可视化、预处理、特征降维、模型训练等常见的高光谱检测功能。下面是对各模块的简要介绍：

## load模块

该模块是HDT工具包的数据加载模块，包括原始光谱数据的批量ROI区域提取，原始光谱矩阵的保存和加载，以及数据清洗的功能。

- `region_of_interest_extraction()`函数实现了利用二值化和阈值分割实现感兴趣区域提取的功能。
- `load_hyperspectral_table()`实现了读取光谱矩阵表格的功能。

## preprocess模块

该模块是HDT工具包的数据预处理模块，包括ROI区域提取、光谱曲线预处理等功能。

- `largest_connected_component_cropping()`函数实现了ROI区域提取中，最大连通域裁剪的功能。
- `preprocess_hyperspectral_table()`函数实现了光谱矩阵表格预处理的功能。

## feature_dimension_reduction模块

该模块是HDT工具包的特征降维模块，包括竞争自适应重加权采样算法、连续投影法。

- `competitive_adapative_reweighted_sampling()`函数实现了基于竞争自适应重加权采样算法的特征选择功能。
- `successive_projections_algorithm()`函数实现了基于连续投影法的特征选择功能。

## split模块

该模块是HDT工具包的数据划分模块，包括SPXY算法。

- `sample_partitioning_based_on_joint_x_y_distance()`函数实现了基于SPXY算法（sample set partitioning based on joint x-y distance）的样本划分功能。

## train模块

该模块是HDT工具包的模型训练模块。

- `train_spectral_table()`函数实现了对光谱表格数据进行多重随机划分或SPXY划分训练模型的功能。

# 程序示例

[Hyperspectral_Detection_Tools | HDT高光谱检测工具使用说明](https://www.bilibili.com/video/BV18u4m1A7Fw/)

首先是包的安装，目前1.0版本只有手动安装模式，需要提前下载包的tar.gz文件。

下载完成后，在终端激活虚拟环境（如果有的话），使用cd命令进入到包文件所在的路径。

之后使用`pip install 包文件名`的命令安装包。

---

现在是感兴趣区域提取功能的使用说明，首先导入模块并为模块设置别名为hdt。

```python
import Hyperspectral_Detection_Tools as hdt
```

随后使用hdt包中load模块的`region_of_interest_extraction()`方法实现感兴趣区域提取。

```python
hdt.load.region_of_interest_extraction(
    rootPath="Hyperspectral_Data/rawPath",
    targetPath="Hyperspectral_Data/targetPath.csv"
)
```

rootPath参数对应的字符串是原始光谱图片数据的路径，它是一个文件夹，文件夹下每个样本都以编号命名，都有对应的hdr文件和raw文件。

targetPath参数对应的字符串是预测数据表格的路径，它是一个csv文件，文件第1列是与rootPath对应的样本编号，第2列是每个样本对应的目标元素含量。

经过一段时间的等待，感兴趣区域提取顺利完成，该函数的返回结果是一个DataFrame表格。

该表格第1列是样本编号，第2列到倒数第2列是各波段的光谱反射率，最后1列是每个样本编号对应的预测数据，如果targetPath的csv没有则为Nan。

该表格可以保存以待下一步的分析。

---

现在是数据读取、预处理、模型训练、数据可视化等功能的使用说明。

刚才保存的csv文件可以使用hdt包中load模块的`load_hyperspectral_table()`函数读取。

```python
W, X, y = hdt.load.load_hyperspectral_table(
    csvPath="Test.csv",
    spectral_indexes=[1, 177],
    target_indexes=[-1]
)
```

该表格共有273行（即273条数据），178列（即176个波段）。

因此spectral_indexes参数设置为[1,177]，表示取索引为1列到索引为177的列作为光谱数据（左闭右开）。

而target_indexes设为[-1]表示取倒数第1列作为目标数据。

可以使用hdt包中plot模块的`plot_hyperspectral_curve()`函数可视化读取的数据，其中X是光谱数据，W是各波段的波长列表。

```python
hdt.plot.plot_hyperspectral_curve(
    hyperspectral_data=X,
    wave=W
)
```

可以使用hdt包中preprocess模块的`preprocess_hyperspectral_table()`函数预处理光谱数据，使用preprocess_method参数指定预处理方法为SNV。

```python
X_SNV = hdt.preprocess.preprocess_hyperspectral_table(
    origin_hyperspectral_data=X,
    preprocess_method="SNV"
)
```

同样可以使用plot_hyperspectral_curve()方法可视化SNV预处理后的光谱曲线数据。

```python
hdt.plot.plot_hyperspectral_curve(
    hyperspectral_data=X_SNV,
    wave=W
)
```

---

最后，可以使用hdt中train模块的`train_spectral_table()`方法训练模型，使用split_method指定划分方法为spxy，使用test_size参数指定测试集比例为0.3，使用n_iter参数指定随机搜索模型参数次数为2000次，使用model参数指定模型为KRR。

经过一段时间的等待，模型训练完成，返回一个包括随机数种子、模型参数、模型表现的字典。

```python
  0%|          | 0/1 [00:30<?, ?it/s]

{'random_state_split': 'spxy',
 'random_state_search': 1059373329,
 'best_params': {'alpha': 0.013905845949127471,
  'gamma': 0.561236880728405,
  'kernel': 'rbf'},
 'train_r2_full': 0.6724547475007336,
 'train_rmse_full': 0.17314872949125226,
 'test_r2_full': 0.7068965955578606,
 'test_rmse_full': 0.16381845989198573,
 'train_rpd_full': 1.7518801822369052,
 'test_rpd_full': 1.216923315562316,
 'feature_number': 176}
```

# 未来计划

- [ ] 完善说明文档。
- [ ] 优化函数代码。
- [ ] 添加特征降维、可视化模块新函数。