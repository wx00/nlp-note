import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, average_precision_score

from week1 import feat_base


def train():
    meta = pd.read_pickle(feat_base + 'train-meta').iloc[0]
    batch_num, batch_size = int(meta['batch_num']), int(meta['batch_size'])
    feat_fmt, label_fmt = meta['feat_fmt'], meta['label_fmt']

    model = LogisticRegression(penalty='l1', max_iter=200)

    print('training started')
    for i in range(batch_num):
        print('train epoch {0}'.format(i+1))
        samples = pd.read_pickle(feat_base + feat_fmt.format(i)).to_coo().tocsr()
        targets = np.array(pd.read_pickle(feat_base + label_fmt.format(i)))
        model.fit(samples, targets)
        del samples, targets
    print('training finished')

    model.sparsify()
    joblib.dump(model, feat_base + 'model.pickle')


def validate():
    vocabulary = np.load(feat_base + 'dict.npy')
    model = joblib.load(feat_base + 'model.pickle')
    weight = model.coef_.toarray().reshape((-1,))
    print(weight.shape)
    index = weight.argsort()[::-1]
    best5 = zip(vocabulary[index[0:5]], weight[index[0:5]])
    worst5 = zip(vocabulary[index[-5:]], weight[index[-5:]])
    print(list(best5))
    print(list(worst5)[::-1])
    del index, vocabulary

    meta = pd.read_pickle(feat_base + 'test-meta').iloc[0]
    batch_num, batch_size = int(meta['batch_num']), int(meta['batch_size'])
    feat_fmt, label_fmt = meta['feat_fmt'], meta['label_fmt']

    score = []
    print('test started')
    for i in range(batch_num):
        print('test epoch {0}'.format(i + 1))
        samples = pd.read_pickle(feat_base + feat_fmt.format(i)).to_coo().tocsr()
        targets = np.array(pd.read_pickle(feat_base + label_fmt.format(i)))
        predict = model.predict(samples)
        score.append((len(predict), accuracy_score(targets, predict), recall_score(targets, predict)))
        del samples, targets
    print('test finished')

    print(score)
    num, accuracy, recall = zip(*score)
    num = np.array(num)
    w = num / num.sum()
    print('accuracy : {0}'.format((np.array(accuracy) * w).sum()))
    print('recall : {0}'.format((np.array(recall) * w).sum()))


if __name__ == '__main__':
    # train()
    validate()
