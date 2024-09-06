import numpy as np
class Perceptron:
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])  #生成一个符合0附近正态分布的随机数数组，赋值给权重数组w_，X的维度就是w的维度
        self.b_ = np.float64(0.)    #把偏置b设置为numpy内置的浮点数类型
        self.errors_ = []   #列表，统计每次训练产生的预测错误次数
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
            #zip函数用于把多个可迭代对象压缩在一起，生成一个由元组组成的迭代器，元素位置和声明时一样，可以*并行遍历*
            #这里xi就是训练集的对象，target则是训练集的类标签
                update = self.eta * (target - self.predict(xi))  #predict函数的实现在后面
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)  #当每次发生更新时，说明预测出现了错误，递增errors
            self.errors_.append(errors)   #把每次预测错误次数errors置入列表
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        #where是np中的条件函数，np.where(condition, [x, y]) 代表若条件为True，结果为x；反之为y。这里根据净输入是否大于0分成类1和类0
