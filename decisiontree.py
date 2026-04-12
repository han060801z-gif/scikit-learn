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

from sklearn.metrics import accuracy_score, classification_report

#2.1决策树
from sklearn.tree import DecisionTreeClassifier
#调入决策树模型
clf = DecisionTreeClassifier(max_depth = 5,min_samples_split=2, min_samples_leaf=2,random_state=42)
#设置决策树参数：max_depth是决策树的最大深度，即从一开始往下最多能分几层（问几个问题），控制树的复杂度，防止过度学习，默认为none
#min_sample_split:限制节点再划分所需的最小样本数，即一个节点至少要有多少个样本才允许继续分裂，控制过拟合，默认为2
#min_samples_leaf:叶子节点（指树分到最后，不再往下分的节点）的最小样本数，控制过拟合，默认为1，增加可防止模型对噪声过于敏感
#random_state：固定随机种子，让每次运行结果一致
clf.fit(X_train, y_train)
#在训练集上训练决策树模型
y_pred = clf.predict(X_test)
#用训练好的clf模型在验证集上进行预测
accuracy = accuracy_score(y_test, y_pred)
print(accuracy) #计算准确率
print("\n决策树分类报告")
print(classification_report(y_test, y_pred))

#2.2混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   #导入用来计算混淆矩阵的confusion_matrix函数，用来画混淆矩阵的confusionmatrixdisplay
import matplotlib.pyplot as plt  #导入画图工具并简写成plt
cm = confusion_matrix(y_test, y_pred)  #把真实的标签和预测的标签拿来对比，计算出混淆矩阵，存在cm里
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  #创建一个混淆矩阵显示器对象
print(cm) #打印混淆矩阵
disp.plot(cmap="Blues")   #把混淆矩阵用plot画出来，使用蓝色系颜色
          #cmap是colormap的缩写，即选择配色  Blues是指渐变蓝色
plt.show()

#2.3k折交叉验证
#指将数据集均匀划分为k个子集，每个子集都有机会作为验证集，其余作为训练集，看k次的平均值和方差来衡量模型在该交叉验证中的整体性能和稳定性，一般为5或10
#即换一下数据，再训练k次不一样的模型
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=1)
cross_val_scores=cross_val_score(dt_model, X_train, y_train, cv=10)  #进行交叉验证，返回十个验证准确率的数组
x_values=range(1,11)  #定义x轴的数据，即交叉验证的折数

plt.figure(figsize=(8, 6))
plt.plot(x_values, cross_val_scores, marker='o', linestyle='-')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross Validation Scores')
plt.xticks(x_values)
plt.grid(True)
plt.show()

print("这十次的准确率分别是：",cross_val_scores)
print("这十次准确率的均值是",cross_val_scores.mean())