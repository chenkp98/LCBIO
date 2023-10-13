import numpy as np
from scipy import stats
import pandas as pd

def exact_test_threshold(feature_matrix, labels, threshold):
    num_features = feature_matrix.shape[1]
    selected_features = []

    for i in range(num_features):
        feature_values = feature_matrix[:, i]
        contingency_table = np.array([[np.sum((feature_values == 1) & (labels == 1)), np.sum((feature_values == 1) & (labels == 0))],
                                      [np.sum((feature_values == 0) & (labels == 1)), np.sum((feature_values == 0) & (labels == 0))]])
        _, p_value = stats.fisher_exact(contingency_table)

        if p_value < threshold:
            selected_features.append(i)


    return selected_features

# 示例数据
data= pd.read_excel(r"D:\data\生信\1.xlsx")
feature_matrix=data.iloc[:,:-1].values
labels = data.iloc[:, -1].values

# 设定阈值并筛选特征
threshold = 1
selected_features = exact_test_threshold(feature_matrix, labels, threshold)

df = data.iloc[:, selected_features]
df.to_excel("fisher.xlsx", index=False)

print("选择的特征索引:", selected_features)