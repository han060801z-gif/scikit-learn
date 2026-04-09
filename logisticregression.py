import pandas as pd
df = pd.read_excel('wine_data.xlsx')
print("数据形状（行，列）:",df.shape)
print("数值列统计摘要：")
print(df.describe())

#划分目标向量和特征向量
x = df.drop('target', axis=1)#特征矩阵  意思是把df里的target一列删掉，剩下的作为特征矩阵X，axis=0表示按行操作，axis=1表示按列操作
y = df['target']#目标向量，是想预测的结果，即拿X来学怎么预测y

from sklearn.model_selection import train_test_split   #这个函数的作用：把数据随机分成两部分，一部分给模型学习，一部分检验模型学的怎么样，即训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#同时拆分x和y，把20%的数据分给测试集，固定随即划分的方式为42（什么数都行，只需要固定）

from sklearn.metrics import accuracy_score, classification_report #导入准确率和分类报告指标

#1.逻辑回归（logistic regression)
#1.1训练模型
from sklearn.linear_model import LogisticRegression  #导入模型：逻辑回归模型，通常用于二分类、多分类
model = LogisticRegression(max_iter=5000)  #初始化模型：构建一个逻辑回归模型并赋给变量model，iter为迭代次数，意思是模型最多迭代五千次，让模型学习的次数
model.fit(X_train, y_train)  #在训练集上训练模型，传进训练集的特征和真实标签，进而学习到哪些特征的x会得到什么样的y
y_pred = model.predict(X_test)   #在测试集上做预测。让训练好的模型对X_test做预测，把预测结果保存到y_pred里
accuracy = accuracy_score(y_test, y_pred)  #计算准确率，用accuracy_score函数
print("逻辑回归模型准确率：",accuracy)
print("\n逻辑回归分类报告")
print(classification_report(y_test, y_pred)) #这是一个用于分类评估的函数，可以输出精确率、召回率等指标

#1.2混淆矩阵检验：更细致的看模型到底哪里分对了哪里分错了
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   #导入用来计算混淆矩阵的confusion_matrix函数，用来画混淆矩阵的confusionmatrixdisplay
import matplotlib.pyplot as plt  #导入画图工具并简写成plt
cm = confusion_matrix(y_test, y_pred)  #把真实的标签和预测的标签拿来对比，计算出混淆矩阵，存在cm里
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  #创建一个混淆矩阵显示器对象
print(cm) #打印混淆矩阵
disp.plot(cmap="Blues")   #把混淆矩阵用plot画出来，使用蓝色系颜色
          #cmap是colormap的缩写，即选择配色  Blues是指渐变蓝色
plt.show()
