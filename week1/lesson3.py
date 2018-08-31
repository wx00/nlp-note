
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

prof = "train"
# neg = "/Users/pengshuolin/Downloads/aclImdb/{0}/neg/".format(prof)
# pos = "/Users/pengshuolin/Downloads/aclImdb/{0}/pos/".format(prof)
db = "/Users/pengshuolin/data/{0}/".format(prof)

# print(os.listdir(neg))
# print(os.listdir(pos))
#
# neg_db = []
# for n in os.listdir(neg):
#     with open(neg + n) as f:
#         neg_db.append(f.read())
#
# pos_db = []
# for p in os.listdir(pos):
#     with open(pos + p) as f:
#         pos_db.append(f.read())
#
# np.save(db + "neg.npy", neg_db)
# np.save(db + "pos.npy", pos_db)

neg_db = np.load(db + "neg.npy")
pos_db = np.load(db + "pos.npy")
encoder = TfidfVectorizer(min_df=5, max_df=.5)
features = encoder.fit_transform(np.concatenate((neg_db, pos_db)))
pd.DataFrame(features.todense(), columns=encoder.get_feature_names()).to_pickle(db + "{0}-data".format(prof))

"""
text classification：文本分类模型

sentiment classification: 情感分类
"""