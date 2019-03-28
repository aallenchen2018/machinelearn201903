import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch

word_level_file='./data/word-level/train.jieba.bat'
word_tfidf_file='./data/word-level/train.tfidf.bat'

#读文件
def _read_file(filepath):
    with open(filepath,'rb') as f:
        bunch = pickle.load(f)
    return bunch
#写文件
def _wirte_file(filepath,bunch):
    with open(filepath,'wb') as f:
        pickle.dump(bunch,f)


#####获取停用词 
def get_stop_words(filepath='./data/中文停用词库.txt'):
    stop_words=[]
    for line in open(filepath,'r',encoding='gb18030'):
        stop_words.append(line.strip())
    return stop_words

####tfidf量化文章信息
def gen_tfidf(inputfile,output):
    bunch = _read_file(inputfile)
    tfidf_bunch=Bunch(category_labels={},labels=[],tfidf=[],vocabulary={})
    tfidf_bunch.category_labels=bunch.category_label#类别和标签对应的字典
    tfidf_bunch.labels = bunch.labels#文章的标签
    stop_words=get_stop_words()
    tfidf = TfidfVectorizer(stop_words=stop_words,sublinear_tf=True,max_df=0.8)
    tfidf_bunch.tfidf=tfidf.fit_transform(bunch.contents)#tfidf转化后文章信息
    tfidf_bunch.vocabulary = tfidf.vocabulary_ #词汇对照字典
    #保存信息
    _wirte_file(output,tfidf_bunch)

gen_tfidf(word_level_file,word_tfidf_file)
