import pandas as pd 
import numpy as np  
from matplotlib import pyplot as plt 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score 

data=pd.read_csv('/home/aallen/git/Mushroom/data/mushrooms.csv')
data.head() 

data.info() 
data['class'].unique()
data['class'].value_counts() 
data.shape 

######特征编码 
pd.get_dummies(data[['cap-color','bruises']]) #哑变量,Onehot编码 

import pandas as pd  
from sklearn.preprocessing import LabelEncoder 

a=pd.get_dummies(data.drop(['class'],axis=1))  #独热编码/虚拟变量/哑变量 
encoder = LabelEncoder()  #标签编码 
a['class']=encoder.fit_transform(data['class'])   ###在这一列把class转成独热
a.head()

y=a['class']
X=a.drop('class',axis=1) 
X

####数据集是一个文件，我们自己分出一部分来做测试吧（不是校验集） 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
columns=X_train.columns 
columns




######default Logistic Regression 
from sklearn.linear_model import LogisticRegression 
model_LR=LogisticRegression() 
model_LR.fit(X_train,y_train) 

##看看各特征的系数,系数的绝对值大小可视为该特征的重要性 
fs=pd.DataFrame({'columns':list(columns),'coef':list(abs(model_LR.coef_.T))})
print(fs.sort_values(by=['coef'],ascending=False))

#计算精确度
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
print('The accuary of default Logistic Regression is',model_LR.score(X_test, y_pred))
#计算ROC的面积AUC
print('The AUC of default Logistic Regression is', roc_auc_score(y_test,y_pred))


#####Logistic Regression(Tuned model) 
from sklearn.linear_model import LogisticRegression 
LR_model=LogisticRegression() 

#设置参数搜索范围(Grid,网格) 
tuned_parameters={'C':[0.001,0.01,0.1,1,10,100,1000],'penalty':['l1','l2']} 

#####CV  
#fit函数执行会优点慢,因为要循环执行   参数数目*CV折数 次模型 训练
LR=GridSearchCV(LR_model,tuned_parameters,cv=10) 
LR.fit(X_train,y_train) 
print(LR.best_params_) 

y_prob = LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR.score(X_test, y_pred)#准确率

print('The AUC of GridSearchCV Logistic Regression is', roc_auc_score(y_test,y_pred))

##根据CV得到的结果C=1,penalty='l1' 为最好,代入LogisticRegression 
LR_model1= LogisticRegression(C=1,penalty='l1')
LR_model1.fit(X_train,y_train)

#看看各特征的系数,系数的绝对值大小可视为该特征的重要性 
fs1 = pd.DataFrame({"columns":list(columns), "coef":list(abs(LR_model1.coef_.T))})
fs1.sort_values(by=['coef'],ascending=False)  ##可以print!! >>print(fs1.sort_values(by=['coef'],ascending=False))
(fs1['coef']>0).sum() 

######default Decision tree model  
from sklearn.tree import DecisionTreeClassifier 
model_tree=DecisionTreeClassifier() 
model_tree.fit(X_train,y_train) 
#De..的参数>>>>
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')
y_prob=model_tree.predict_proba(X_test)[:,1]  #
y_pred=np.where(y_prob>0.5,1,0) 
model_tree.score(X_test,y_pred) 
print('The AUC of default Desicion Tree is', roc_auc_score(y_test,y_pred))
df = pd.DataFrame({"columns":list(columns), "importance":list(model_tree.feature_importances_.T)})
df.sort_values(by=['importance'],ascending=False)  #可打印 

(df['importance']>0).sum() 
##图形显示那些特征重要性高
plt.bar(range(len(model_tree.feature_importances_)), model_tree.feature_importances_)
plt.show()

####可根据特征重要性做特征选择 
from numpy import sort 
from sklearn.feature_selection import SelectFromModel 

thresholds=sort(model_tree.feature_importances_) 
for thresh in thresholds:
    #select features using threshold 
    selection=SelectFromModel(model_tree,threshold=thresh,prefit=True) 
    select_X_train=selection.transform(X_train) 
    #train model  跑train
    selection_model=DecisionTreeClassifier() 
    selection_model.fit(select_X_train,y_train) 

#eval model 这里应该是跑test
    select_X_test=selection.transform(X_test) 
    y_pred=selection_model.predict(select_X_test)
    predictions=[round(value) for value in y_pred]
    accuracy=accuracy_score(y_test,predictions) 
    
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],accuracy*100.0))
######上面这一步是先跑训练再跑测试,得到一个结果(特征的对结果的影响度,和精确度)


##Fit model using the best threshhold 
thresh =0.006 
selection=SelectFromModel(model_tree,threshold=thresh,prefit=True) 
select_X_train=selection.transform(X_train) 
# train model
selection_model = DecisionTreeClassifier()
selection_model.fit(select_X_train, y_train)

select_X_test= X_test[df[df['importance']>0.006]['columns']]#对测试集做同样的处理
##精准度 
y_pred = selection_model.predict(select_X_test) # This will give you positive class prediction probabilities  
selection_model.score(select_X_test, y_pred)


#####决策树的可视化
from sklearn import tree
tree.export_graphviz(selection_model, out_file='best_tree.dot') 

from sklearn import tree
from IPython.display import display, Image
import pydotplus
dot_data = tree.export_graphviz(model_tree, 
                                out_file=None, 
                                feature_names=columns,
                                class_names = ['p', 'e'],
                                filled = True,
                                rounded =False
                               )
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))
graph = pydotplus.graph_from_dot_data(dot_data)  # doctest: +SKIP
graph.write_pdf('123.pdf')