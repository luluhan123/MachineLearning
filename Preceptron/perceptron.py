import numpy as np
import copy

class Preceptron(object):
    #初始化参数，eta是学习率
    def __init__(self, x_train, y_train, eta = 1, w0 = 0, b0 = 0, alpha = 0):
        self.eta = eta
        self.b = b0
        self.w = np.array([w0 for i in range(x_train.shape[1])])
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = np.array([alpha for i in range(x_train.shape[0])])
        self.iter_matrix = []

    #学习算法原始形式
    def orig_iter(self):
        adjust_flag = False
        iter_number = 0

        for x,y in zip(self.x_train, self.y_train):
            iter_number += 1

            while y * (np.dot(self.w,x) + self.b) <= 0:
                adjust_flag = True
                self.w += self.eta * y * x
                self.b += self.eta * y

                self.iter_matrix.append([x, copy.copy(self.w),self.b])

            if iter_number != 1 and adjust_flag:
                adjust_flag = False
                self.orig_iter()

    # 计算gram矩阵
    def gram(self,x):
        gram_matrix = [[] for i in range(x.shape[0])]
        for index, i in enumerate(x):
            for j in x:
                gram_matrix[index].append(np.dot(i, j))
        return np.array(gram_matrix)


    def duality_condition(self, index, gram_m):
        sum_temp = 0
        for i in range(self.x_train.shape[0]):
            sum_temp += self.alpha[i] * self.y_train[i] * gram_m[index][i]
        return self.y_train[index] * (sum_temp + self.b)

    def duality_iter(self, gram_m):
        # 对偶学习算法
        adjust_flag = False
        iter_number = 0

        for i in range(self.x_train.shape[0]):
            iter_number += 1
            while self.duality_condition(i, gram_m) <= 0:
                adjust_flag = True
                self.alpha[i] += 1
                self.b += self.y_train[i]

                self.iter_matrix.append([self.x_train[i], self.compute_wb()])

            if iter_number != 1 and adjust_flag:
                adjust_flag = False
                self.duality_iter(gram_m)

    def duality_study(self):
        gram_m = self.gram(self.x_train)
        self.duality_iter(gram_m)

        return self.compute_wb()

    def compute_wb(self):
        w = 0
        b = 0
        for i in range(self.x_train.shape[0]):
            w += self.alpha[i] * self.y_train[i] * self.x_train[i]
            b += self.alpha[i] * self.y_train[i]
        return w, b

    def pprint(self):
        print('误分类点   w   b ')
        for i in self.iter_matrix:
            print(i)

if __name__ == '__main__':
    data_x = [[3, 3], [1, 1], [4, 3]]
    data_y = [1, -1, 1]
    data_x, data_y = np.array(data_x), np.array(data_y)

    a = Preceptron(data_x, data_y)
    a.orig_iter()
    a.pprint()
    print("-----------------------------------------------")
    a.duality_study()
    a.pprint()
    print("-----------------------------------------------")
    print(a.iter_matrix[-1][1][0])