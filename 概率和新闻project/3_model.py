import pickle
import os
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB

word_tfidf_file='./data/word-level/train.tfidf.bat'
#读文件
def _read_file(filepath):
    with open(filepath,'rb') as f:
        bunch = pickle.load(f)
    return bunch

bunch = _read_file(word_tfidf_file)#获取文件内的信息

X=bunch.tfidf#提取数据集的特征
print(X.shape)
print(type(X))
y=bunch.labels#提取数据集的标签
print(len(y))
print(y[:10])

#####拆分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

######建立模型
nb=MultinomialNB(alpha=0.01)   ###alpha为平滑参数 默认是1 
nb.fit(X_train,y_train)#训练模型

from sklearn.metrics import classification_report

y_pred  = nb.predict(X_test)
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix#混淆矩阵
print(confusion_matrix(y_test,y_pred))

#####参数调整 

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
alphas=[0.001,0.01,0.1,1]
for alpha in alphas:
    nb1=MultinomialNB(alpha=alpha)
    nb1.fit(X_train,y_train)
    y_pred1  = nb1.predict(X_test)
    precision=precision_score(y_test,y_pred1,average='weighted')
    recall = recall_score(y_test,y_pred1,average='weighted')
    print(alpha,precision,recall)




#######进一步优化  1, max限制放宽. 2,ngram_range 修改 TfidfVectorizer中这个参数,
########记录下相邻词的先后顺序, 一般可以是2或者3,但是要求电脑高配置
