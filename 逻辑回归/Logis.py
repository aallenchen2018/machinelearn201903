import pandas as pd 
import numpy as np 
from sklearn.metrics import log_loss 
import matplotlib.pyplot as plt 
import seaborn as sns 


dpath='/home/aallen/git/Logistic 回归——Otto商品分类'
train=pd.read_csv('/home/aallen/git/Logistic 回归——Otto商品分类/Otto_train.csv',index_col=['id'])
# print(train.head())
# print(train.shape)

train['target'].value_counts().plot(kind='bar')
plt.ylabel('各个类别占比')

#####特征编码
y_train = train['target']   #形式为Class_x
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)
train = train.drop('target',axis=1)
X_train = np.array(train)
print(y_train.unique())


######数据预处理
from sklearn.preprocessing import StandardScaler
##初始化特征的标准化器
ss_X=StandardScaler()
#分别对训练和测试数据的特征进行标准化处理
X_train=ss_X.fit_transform(X_train)



######模型训练

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#交叉验证用于评估模型性能和进行参数调优
#分类任务中交叉验证缺省是采用StratifiedkFold
from sklearn.model_selection import cross_val_score 
loss=cross_val_score(lr,X_train,y_train,cv=5,scoring='accuracy')
print(' logloss of each fold is : ', loss )
print('cv logloss is :', loss.mean())


#####正则化的Logistic Regression及参数调优
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 

param_grid={'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}
lr_penalyty=LogisticRegression(class_weight='balanced',solver='saga',max_iter=1000)  ##实例化模型

grid=GridSearchCV(lr_penalyty,param_grid,cv=5,scoring='accuracy')
grid.fit(X_train,y_train)
grid.cv_results_ 

print(grid.best_score_)
print(grid.best_params_)

# ###绘图
# #plot CV 误差曲线
# test_means=grid.cv_results_['mean_test_score']
# train_means=grid.cv_results_['mean_train_score']
# #plot results 
# n_Cs=len(Cs)
# number_penaltys=len(penaltys)
# test_scores=np.array(test_means).reshape(n_Cs,number_penaltys)
# train_scores=np.array(train_means).reshape(n_Cs,number_penaltys)

# x_axis = np.log10(Cs)
# for i, value in enumerate(penaltys):
#     #pyplot.plot(log(Cs), test_scores[i], label= 'penalty:'   + str(value))
#     pyplot.errorbar(x_axis, test_scores[:,i], yerr=test_stds[:,i] ,label = penaltys[i] +' Test')
#     pyplot.errorbar(x_axis, train_scores[:,i], yerr=train_stds[:,i] ,label = penaltys[i] +' Train')
    
# pyplot.legend()
# pyplot.xlabel( 'log(C)' )                                                                                                      
# pyplot.ylabel( 'loss' )
# pyplot.savefig('LogisticGridSearchCV_C.png' )

# pyplot.show()











