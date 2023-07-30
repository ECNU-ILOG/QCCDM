
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

train_data = pd.read_csv('train.csv').to_numpy()
valid_data = pd.read_csv('valid.csv').to_numpy()
test_data = pd.read_csv('test.csv').to_numpy()
s = np.vstack([train_data, valid_data, test_data])
idx = np.lexsort((s[:, 1], s[:, 0]))

# 按照排序结果重新排列数组
a_sorted = s[idx]
np.savetxt('a910TotalData.csv', a_sorted, delimiter=',')
print(a_sorted.shape)

# r_matrix = -1 * np.ones(shape=(4163, 17746))
# train_data = pd.read_csv('train.csv').to_numpy()
# test_data = pd.read_csv('test.csv').to_numpy()
# valid_data = pd.read_csv('valid.csv').to_numpy()
# for i in range(train_data.shape[0]):
#     s = train_data[i, 0] - 1
#     p = train_data[i, 1] - 1
#     score = train_data[i, 2]
#     r_matrix[s, p] = score
#
# for i in range(test_data.shape[0]):
#     s = int(test_data[i, 0]) - 1
#     p = int(test_data[i, 1]) - 1
#     score = test_data[i, 2]
#     r_matrix[s, p] = int(score)
#
# for i in range(valid_data.shape[0]):
#     s = int(valid_data[i, 0]) - 1
#     p = int(valid_data[i, 1]) - 1
#     score = valid_data[i, 2]
#     r_matrix[s, p] = score
# np.savetxt('a910rmatrix.csv', r_matrix, delimiter=',')
