import nltk


# transform token into feature


# bag of words 词袋模型
# text vectorization 文本向量化
"""
n-gram n元模型 （token 的个数）

出现频率：
高频：停止词（去除）
低频：拼写错误（去除）
中频：好的模式

"""

"""
TF-IDF

当前文档的权重
term (n-grams) frequency in document: tf(term, doc)
binary: 0,1
raw count: count(term, doc)
term freq: count(term, doc) / length(doc)
log norm: 1+ log(count(term, doc))

1 / 在整个语料库中的权重
inverse document frequency ：
D : corpus of documents 
N = |D| : total number of document
appear(t) = |{ count(t) in D : count(d) in D where t in d }|: num of doc where term t appears
idf(t, D) : log(N/appear(t))

TF-IDF： 在当前文档的权重 / 在整个数据集的权重 
tfidf(t,d,D) = tf(t,d) * idf(t, D)

"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

texts = [
    "good movie",
    "not a good movie",
    "did not like",
    "i like it",
    "good one"]
tfidf = TfidfVectorizer(min_df=2, max_df=.5, ngram_range=(1, 2))
features = tfidf.fit_transform(texts)

print(
    pd.DataFrame(features.todense(), columns=tfidf.get_feature_names()))