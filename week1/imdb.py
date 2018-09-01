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


def shuffle_feat(feat_pos, feat_neg, batch_size, feat_repo, prefix):

    num_pos, num_neg = feat_pos.shape[0], feat_neg.shape[0]
    total_num = num_pos + num_neg
    batch_size = min(total_num, batch_size)

    ratio = num_pos / total_num
    batch_pos = int(ratio * batch_size)
    batch_neg = batch_size - batch_pos

    batch_num = total_num // batch_size
    assert num_pos // batch_pos == num_neg // batch_neg == batch_num

    feat_fmt, label_fmt = prefix + '-feat-{0}', prefix + '-label-{0}'
    for i in range(batch_num):
        feat_name, label_name = feat_fmt.format(i), label_fmt.format(i)

        s_pos = slice(i * batch_pos, min((i+1)*batch_pos, num_pos))
        s_neg = slice(i * batch_neg, min((i+1)*batch_neg, num_neg))
        pos, neg = pd.SparseDataFrame(feat_pos[s_pos], default_fill_value=.0), \
                   pd.SparseDataFrame(feat_neg[s_neg], default_fill_value=.0)
        pd.concat((pos, neg)).astype(np.float32).to_pickle(feat_repo + feat_name)
        pos_len, neg_len = len(pos), len(neg)
        del pos, neg

        pos, neg = pd.Series(np.ones(pos_len)), pd.SparseSeries(np.zeros(neg_len))
        pd.concat((pos, neg)).to_pickle(feat_repo + label_name)
        del pos, neg

        print("batch[{0}]: {1}-{2} {3}-{4}".format(
            i, s_pos.start, s_pos.stop, s_neg.start, s_neg.stop))

    pd.DataFrame(
        data=np.array([[batch_num, batch_size, feat_fmt, label_fmt]]),
        columns=('batch_num', 'batch_size', 'feat_fmt', 'label_fmt')
    ).to_pickle(feat_repo + prefix + '-meta')


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
                              ngram_range=(1, 2 if bi_gram else 1),
                              tokenizer=tokenizer if custom_tokenizer else None)
    features = encoder.fit_transform(raw_doc)
    vocabulary = encoder.get_feature_names()

    np.save(feat_base + 'dict.npy', vocabulary)

    shuffle_feat(features[train_pos_range], features[train_neg_range],
                 3000, feat_base, 'train')

    shuffle_feat(features[test_pos_range], features[test_neg_range],
                 5000, feat_base, 'test')

    # for n, s in {'train_pos': train_pos_range,
    #              'train_neg': train_neg_range,
    #              'test_pos': test_pos_range,
    #              'test_neg': test_neg_range}.items():
    #     print('saving {0}'.format(n))
    #     mat = pd.SparseDataFrame(features[s], columns=encoder.get_feature_names())
    #     mat.to_csv(feat_base + '{0}.csv'.format(n))  # mat.to_pickle(feat_base + n)
    #     del mat
    #     print('done {0}'.format(n))


if __name__ == '__main__':
    extract_feat(bi_gram=False)
    meta = pd.read_pickle(feat_base + 'train-meta')
    print(meta['batch_num'], meta['batch_size'], meta['feat_fmt'], meta['label_fmt'])

    meta = pd.read_pickle(feat_base + 'test-meta')
    print(meta['batch_num'], meta['batch_size'], meta['feat_fmt'], meta['label_fmt'])

# a = ['00',  '000',  '001', '0a', '0zwi0ck0']
# x = re.compile('^[0-9]+$')
# for i in a:
#     print(i)
#     print(x.match(i))
