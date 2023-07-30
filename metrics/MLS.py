import numpy as np
import pickle


def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))


def MLS(mas_list):
    mls = 0.0
    for i in range(len(mas_list)):
        for j in range(i + 1, len(mas_list)):
            mls += euclidean_distance(mas_list[i], mas_list[j])
    return 2 * mls / len(mas_list) / (len(mas_list) - 1)


for datatype in ['Math1', 'Math2']:
    for method in ['qccdm-softplus-mf',  'qccdm-softplus-single', 'qccdm-sigmoid-mf',  'qccdm-sigmoid-single', 'qccdm-tanh-mf',  'qccdm-tanh-single', 'kancd', 'ncdm']:
        mas_list = []
        for i in range(10):
            if i != 3:
                path = 'D:\Cs\code\Code\work\CDM-ILOG\exps\QCCDM\logs\cdm\{}\{}-{}-seed{}-Mas.pkl'.format(method,
                                                                                                          method,
                                                                                                          datatype, i)
                with open(path, 'rb') as f:
                    mas = pickle.load(f)
                    mas_list.append(mas)
        print(method, datatype, MLS(mas_list))
