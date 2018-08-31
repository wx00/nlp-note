import nltk

# nltk.download('wordnet')

text = "the quick brown foxes suddenly 100 jumps over the lazy dog, do you see?"

# 分词
to = nltk.tokenize.TreebankWordTokenizer()
tokens = to.tokenize(text)

# token normalization

# Stemming
# 词干提取：去除后缀
st = nltk.stem.PorterStemmer()
print([" ".join(st.stem(token) for token in tokens)])


# Lemmatization
# 词型还原：恢复成字典形式
lem = nltk.stem.WordNetLemmatizer()
print([" ".join(lem.lemmatize(token) for token in tokens)])

# import re
# regex = re.compile('^[0-9]+$')
# token = nltk.tokenize.TreebankWordTokenizer()
# lemma = nltk.stem.WordNetLemmatizer()
#
# def tokenizer(text):
#     tokens = token.tokenize(text)
#     return tuple(map(lemma.lemmatize, filter(lambda t: not regex.match(t), tokens)))
#
# print(tokenizer(text))

# capital letter
# 首字母小写化

# acronym
# 首字母缩略词

"""
text is a sequence of tokens
tokenization is a process of extracting token
normalize tokens using stemming or lemmatization or casing and acronyms
"""


