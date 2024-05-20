#### 1.主线逻辑(清晰)
- 目的：证明deepforest模型的优越性
- 途径：对于期货价格预测问题，运用deepforest，DNN，GAN，Treereg进行预测结果与指标的对比
- 预期效果：deepforest的预测效果优于其余三种模型;
- 要求：体现难度深度广度，创新点
- 数据来源：合理清晰；
#### 2.anconda环境
- 打包环境

``conda env export > environment.yml``
- 加载环境

``conda env create -f environment.yml``
### 3.突出点

- 获取模型的最优参数的[随机搜索](DeepForest/TRY.ipynb)策略;
- 对时间序列数据的[滑动窗口处理](DeepForest/TRY.ipynb);
- 对时间日期数据的[时间特征提取](DeepForest/TRY.ipynb);
- 对非数值型特征的[序列编码处理](DeepForest/TRY.ipynb);
- [对时间序列数据的训练集与测试集的划分](DeepForest/TRY.ipynb);
- [将GAN应用于回归问题的方法](GAN/TRY.ipynb);
- [DeepForest的个人理解](DeepForest/EXPLAINATION.md)
- [Treereg的个人理解](Treereg/EXPLAINATION.md)

### 4.完成成事项
- DNN模型未做详细说明
- GAN模型未做详细说明
- 未关联WPS的笔记