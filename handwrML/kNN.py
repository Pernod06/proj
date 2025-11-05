import numpy as np
from caffe2.perfkernels.hp_emblookup_codegen import prefix


class nKK():
    def __init__(self, k, X_train, Y_train, X_test):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.neighbors = np.zeros((len(self.X_test), len(self.X_train)))

    # 欧式距离
    def EuclDsit(self, x0, x1):
        return np.sum(np.square(x1 - x0))

    # 计算当前数据与标签数据距离
    def Allneighbors(self):
        for i in range(len(self.X_test)):
            for j in range(len(self.X_train)):
                self.neighbors[i, j] = self.EuclDsit(self.X_test[i], self.X_train[j])



    # 下标转为类别
    def index2label(self, index):
        knearest = self.Y_train[index][:self.X_test.shape[0]]
        # 统计K近邻的大多数
        predict = []
        for i in range(self.X_test.shape[0]):
            predict.append(np.argmax(np.bincount(knearest[i])))

        return np.array(predict)

    # 下标转为数值
    def index2values(self, index):
        knearest = self.Y_train[index][:self.X_test.shape[0]]

        predict = np.mean(knearest, axis=1)
        return predict.reshape(-1)


    # knn主干
    def kNN(self, mode='classification'):
        self.Allneighbors()
        # 排序
        self.sort_index = np.argsort(self.neighbors, axis=1, kind='quicksort', order=None)
        # 取前k个近邻
        self.sort_index = self.sort_index[:, 0:self.k]
        if mode == 'classification':
            return self.index2label(self.sort_index)
        if mode == 'regression':
            return self.index2values(self.sort_index)

