import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from week1 import log


base = "D:/Data/aclImdb/"


def list_file(prof):
    neg = base + prof + "/neg/"
    pos = base + prof + "/pos/"
    n = [neg + i for i in os.listdir(neg)]
    p = [pos + i for i in os.listdir(pos)]
    return n + p


doc_list = []
raw_files = list_file("train") + list_file("test")

log.info(raw_files)
for f in raw_files:
    with open(f, encoding='utf-8') as ff:
        doc_list.append(ff.read())

# encoder = TfidfVectorizer(min_df=5, max_df=.5)

