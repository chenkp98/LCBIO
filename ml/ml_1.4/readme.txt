ACGT101_ML_1.0
支持文件类型：txt、csv、excel、pickle（行为样本，列为特征，第一列为样本名，第一行为特征名，最后一列为标签）
特征筛选方法：(随机森林+支持向量机)递归特征消除法、L1正则化
版本支持算法：随机森林、支持向量机、K近邻、决策树、逻辑回归、线性判别分析、朴素贝叶斯、XGBoost、GBDT、ADABoost
支持结果图像：ROC曲线、PR曲线、特征筛选曲线图、L1正则化特征重要性图、混淆矩阵、学习曲线
不支持自动调参，需根据需求手动调参
使用：python.exe  /path/to/ACGT101_ML_1.0/ACGT101_ML_1.0.py -p /path/to/ACGT101_ML_1.0/parameters.txt