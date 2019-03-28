import os
import jieba
import pickle
from sklearn.datasets.base import Bunch#类似字典的数据结构

train_file = 'data/char-level/cnews.train.txt'
test_file = 'data/char-level/cnews.test.txt'
val_file = 'data/char-level/cnews.val.txt'
category_file = 'data/char-level/cnews.category.txt'#类别的信息

new_file='./data/word-level/'
word_level_file='./data/word-level/train.jieba.bat'

if not os.path.exists(new_file):
    os.mkdir(new_file)

class Categories:
    def __init__(self,path_file):
        self.category_label={}#类别和标签对应的字典
        for line in open(path_file,'r',encoding='utf-8'):
            category,label=line.strip().split('\t')
            self.category_label[category]=label
            
    def category_to_label(self,category):#获取对应类别的标签
        return self.category_label[category]

c=Categories(category_file)
print(c.category_to_label('财经'))

#####处理文本信息

def generate_word(inputfilelist,outputfile):
    '''
    处理文章的内容和类别：
    内容部分：通过结巴分词，组织成英语的形式；
    类别部分：转化为类别标签
    并将处理的结果进行保存
    '''
    bunch = Bunch(category_label={},labels=[],contents=[])#声明Bunch包含的信息和类型
    categories=Categories(category_file)#调用类别处理对象
    bunch.category_label=categories.category_label#把类别和标签赋值bunch.category_label
    for inputfile in inputfilelist:
        print('开始处理'+inputfile)
        i=0
        with open(inputfile,'r',encoding='utf-8') as f:
            lines = f.readlines()
        #处理文章和类别
        for line in lines:
            words = ''
            category,content=line.strip().split('\t')
            label=categories.category_to_label(category)#把文章类别转化为标签
            bunch.labels.append(label)
            word_list=jieba.cut(content.strip())#结巴分词
            for word in word_list:
                word = word.strip()
                if word != '':
                    words += word + ' '
            bunch.contents.append(words.strip())#把处理之后的文章内容的信息添加到bunch
            i += 1
        print(i)
    #保存数据
    with open(outputfile,'wb') as fout:
        pickle.dump(bunch,fout)
generate_word([train_file,val_file,test_file],word_level_file)

with open(word_level_file,'rb') as ff:
    bunch = pickle.load(ff)
print(bunch.category_label)

