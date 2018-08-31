import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, average_precision_score

from week1 import feat_base


def train():
    train_pos = pd.read_pickle(feat_base + 'train_pos')
    train_neg = pd.read_pickle(feat_base + 'train_neg')
    samples = np.concatenate((train_pos.values, train_neg.values)).astype(np.float32)
    targets = np.concatenate((np.ones(len(train_pos), dtype=np.int8),
                              np.zeros(len(train_neg), dtype=np.int8)))

    print('training started')
    model = LogisticRegression(penalty='l1', max_iter=200)

    total, batch = len(samples), 3000
    for i in range(0, total, batch):
        s = slice(i, min(i + batch, total))
        model.fit(samples[s], targets[s])

    print('training finished')

    model.sparsify()
    joblib.dump(model, feat_base + 'model.pickle')


def test():
    vocabulary = np.load(feat_base + 'dict.npy')
    print(vocabulary)

    model = joblib.load(feat_base + 'model.pickle')
    weight = model.coef_.reshape((-1,))
    print(weight.shape)
    index = weight.argsort()[::-1]
    best5 = zip(vocabulary[index[0:5]], weight[index[0:5]])
    worst5 = zip(vocabulary[index[-5:]], weight[index[-5:]])
    print(list(best5))
    print(list(worst5)[::-1])

    test_pos = pd.read_pickle(feat_base + 'test_pos')
    test_neg = pd.read_pickle(feat_base + 'test_neg')
    samples = np.concatenate((test_pos.values, test_neg.values))
    targets = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))
    predict = model.predict(samples)

    accuracy = accuracy_score(targets, predict)
    print('accuracy : {0}'.format(accuracy))
    recall = recall_score(targets, predict, (0, 1))
    print('recall : {0}'.format(recall))


if __name__ == '__main__':
    train()
    # test()
