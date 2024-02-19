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

# 未来计划

- [ ] 完善说明文档。
- [ ] 优化函数代码。
- [ ] 添加特征降维、可视化模块新函数。