import os
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from week1 import raw_base, feat_base


def list_file(prof, label):
    d = raw_base + prof + '/{0}/'.format(label)
    return [d + i for i in os.listdir(d)]


def load_doc(prof, label):
    documents = []
    for f in list_file(prof, label):
        with open(f, encoding='utf-8') as ff:
            documents.append(ff.read())
    return documents


def extract_feat(bi_gram=True, custom_tokenizer=False):

    regex = re.compile('^[0-9/]+$')
    token = nltk.tokenize.TreebankWordTokenizer()
    lemma = nltk.stem.WordNetLemmatizer()

    def tokenizer(text):
        tokens = token.tokenize(text)
        return tuple(map(lemma.lemmatize,
                         filter(lambda t: not regex.match(t), tokens)))

    train_pos = load_doc('train', 'pos')
    train_neg = load_doc('train', 'neg')
    test_pos = load_doc('test', 'pos')
    test_neg = load_doc('test', 'neg')

    train_pos_range = slice(0, len(train_pos))
    train_neg_range = slice(train_pos_range.stop, train_pos_range.stop + len(train_neg))
    test_pos_range = slice(train_neg_range.stop, train_neg_range.stop + len(test_pos))
    test_neg_range = slice(test_pos_range.stop, test_pos_range.stop + len(test_neg))
    raw_doc = np.concatenate((train_pos, train_neg, test_pos, test_neg))

    encoder = TfidfVectorizer(min_df=5, max_df=.5, stop_words=stopwords.words(),
                              ngram_range=(1, 2) if bi_gram else None,
                              tokenizer=tokenizer if custom_tokenizer else None)
    features = encoder.fit_transform(raw_doc)
    vocabulary = encoder.get_feature_names()

    np.save(feat_base + 'dict.npy', vocabulary)

    train_pos_mat = pd.SparseDataFrame(
        features[train_pos_range], columns=encoder.get_feature_names(), default_fill_value=.0)
    train_neg_mat = pd.SparseDataFrame(
        features[train_neg_range], columns=encoder.get_feature_names(), default_fill_value=.0)
    test_pos_mat = pd.SparseDataFrame(
        features[test_pos_range], columns=encoder.get_feature_names(), default_fill_value=.0)
    test_neg_mat = pd.SparseDataFrame(
        features[test_neg_range], columns=encoder.get_feature_names(), default_fill_value=.0)

    train_pos_mat.to_pickle(feat_base + 'train_pos')
    train_neg_mat.to_pickle(feat_base + 'train_neg')
    test_pos_mat.to_pickle(feat_base + 'test_pos')
    test_neg_mat.to_pickle(feat_base + 'test_neg')


if __name__ == '__main__':
    extract_feat()
    data = pd.read_pickle(feat_base + 'train_pos')
    print(type(data))
    print(data.shape)
    print(data)

    # train_pos = load_doc('train', 'pos')
    # encoder = TfidfVectorizer(min_df=5, stop_words=stopwords.words())
    # features = encoder.fit_transform(train_pos)
    # vocabulary = encoder.get_feature_names()
    # np.save(feat_base + 'dict.npy', vocabulary)
    #
    # train_pos_mat = pd.SparseDataFrame(
    #     features.todense(), columns=encoder.get_feature_names(), default_fill_value=.0)
    #
    # train_pos_mat.to_pickle(feat_base + 'train_pos')
    #
    # data = pd.read_pickle(feat_base + 'train_pos')
    # print(type(data))
    # print(data.shape)
    # print(data)

    pass

# a = ['00',  '000',  '001', '0a', '0zwi0ck0']
# x = re.compile('^[0-9]+$')
# for i in a:
#     print(i)
#     print(x.match(i))
