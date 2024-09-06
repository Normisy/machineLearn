import os   #提供操作系统交互模块，支持目录操作、进程管理等行为，常用于编写自动化脚本和命令行工具
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #用于管理颜色映射和绘制色图
from Perceptron import Perceptron

def plot_decision_regions(X, y, classifier, resolution = 0.02):  #classifier是我们想研究的分类器，resolution是步长（往后看）
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  #这两个元组在后面绘制边界使用。函数里并未用到
    cmap = ListedColormap(colors[:len(np.unique(y))])   #cmap是一个颜色映射对象，unique(y）得到不同类标签的数量，根据它来决定颜色

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #设置范围：计算X的两个特征的最小值和最大值，适当拓宽1格边界使得图片能完全展现出来

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #np.meshigrid方法生成二维网格数组，作为坐标矩阵，用于绘制决策边界，它接受两个一维数组，合成一个二维网格坐标数组(x1, x2)
    #np.arrange方法生成范围内步长为resolution的数组，我们取步长为0.02（看参数的默认值）

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #np.array().T将网格坐标转换为二维数组，这里xx.ravel()用于把网格点展平为一维数组，方便形成二维数组供预测器使用
    lab = lab.reshape(xx1.shape)
    #将lab的形状调整为与xx1（也是xx2）相同

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    #plt.contourf()方法绘制填充等高线图，前两个参数是网格坐标，第三个是网格点的类别，第四个alpha是透明度，cmap则是颜色映射
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #设置坐标轴显示的范围

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'class {cl}',
                    edgecolor='black')


s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s, header = None, encoding = 'utf-8')
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',0, 1)
#构造y向量:iloc索引器提取数据框中前100行，第5列的数据，用value方法转换为numpy数组
#然后使用numpy的where函数把标签转换为0或1
X = df.iloc[0:100, [0,2]].values
#提取前100行，第1和3列的数据构造X矩阵并转换为numpy形式

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

#用训练好的模型ppn调用可视化函数
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

