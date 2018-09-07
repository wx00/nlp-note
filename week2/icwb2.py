import re
import bisect
import numpy as np
import pandas as pd

from week2 import han_base, raw_base, feat_base, Tag


def read_traditional_alias():
    cht_to_chs = {}
    with open(f'{han_base}/Unihan_Variants.txt') as f:
        line = f.readline()
        while line:
            line = line.strip()
            if line and not line.startswith('#'):
                tokens = line.split()
                if tokens[1] == 'kTraditionalVariant':
                    chs = int(tokens[0][2:], 16)
                    for t in tokens[2:]:
                        cht_to_chs[int(t[2:], 16)] = chs
            line = f.readline()
    return cht_to_chs


def build_vocabulary():
    # cht_to_chs = read_traditional_alias()
    vocabulary = []
    documents = list_training_doc()

    word_set = set()
    for line in documents:
        for token in line:
            for c in token:
                o = ord(c)
                if o not in word_set:
                    word_set.add(o)
                    bisect.insort(vocabulary, o)

    vocabulary = np.array(vocabulary)
    np.save(f'{feat_base}/vocabulary.npy', vocabulary)


def traverse_doc(text, words, tags, converter=None):
    blank = re.compile('\s')
    buff = []

    def flush():
        if words is not None:
            if converter is None:
                words.extend(buff)
            else:
                words.extend(converter(c) for c in buff)
        num = len(buff)
        buff.clear()
        if tags is None:
            return
        if num == 1:
            tags.append(Tag.FIN.value)
        else:
            tags.append(Tag.BEG.value)
            tags.extend(Tag.MID.value for _ in range(num - 2))
            tags.append(Tag.END.value)

    text = text.encode('utf-8').decode('utf-8-sig').strip()
    for w in text:
        if not blank.match(w):
            buff.append(w)
        elif buff:
            flush()

    if len(buff):
        flush()

    if words is not None and tags is not None:
        return words, tags
    if words is not None:
        return words
    return tags


def list_training_doc(tag=False):

    def iter_document(file):
        with open(file, encoding='utf-8') as ff:
            line = ff.readline()
            while line:
                yield traverse_doc(line, [None] if tag else [], [Tag.SEP.value] if tag else None)
                line = ff.readline()

    f = '{0}_training.utf8'
    fmt = f'{raw_base}/training/{f}'
    files = [fmt.format(i) for i in ['as', 'cityu', 'msr', 'pku']]
    return [iter_document(f) for f in files]


def list_test_doc():
    f = '{0}_test.utf8'
    fmt = f'{raw_base}/testing/{f}'
    test = [fmt.format(i) for i in ['as', 'cityu', 'msr', 'pku']]

    f = '{0}_test_gold.utf8'
    fmt = f'{raw_base}/gold/{f}'
    gold = [f'{raw_base}/gold/as_testing_gold.utf8'] + \
           [fmt.format(i) for i in ['cityu', 'msr', 'pku']]

    def iter_document(sample, target):
        with open(sample, encoding='utf-8') as fs, open(target, encoding='utf-8') as ft:
            ls, lt = fs.readline(), ft.readline()
            assert (ls and lt) or (not ls and not lt)
            while ls or lt:
                assert ls and lt
                yield ls, lt
                ls, lt = fs.readline(), ft.readline()

    return [iter_document(t, g) for t, g in zip(test, gold)]


def extract_training_feat(words_per_chunk=2000000, update_file=True):

    vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    word_set = set(vocabulary.tolist())

    def encode(c):
        if c is None:
            return -1
        o = ord(c)
        assert o in word_set
        return np.searchsorted(vocabulary, o)

    chunk_num = 0
    chunk_fmt = 'train-chunk-{0}.npy'
    chunk_name = f'{feat_base}/{chunk_fmt}'

    x_buff, y_buff = [], []
    train_set = list_training_doc(tag=True)
    for doc in train_set:
        for words, tags in doc:
            x_buff.extend(encode(c) for c in words)
            y_buff.extend(tags)
            # print(words)
            # print(tags)
            if len(x_buff) >= words_per_chunk:
                chunk = np.array([x_buff, y_buff])
                if update_file:
                    np.save(chunk_name.format(chunk_num), chunk)
                x_buff.clear()
                y_buff.clear()
                chunk_num += 1

    if len(x_buff):
        chunk = np.array([x_buff, y_buff])
        if update_file:
            np.save(chunk_name.format(chunk_num), chunk)
        chunk_num += 1

    if update_file:
        pd.DataFrame(
            data=np.array([[chunk_num, chunk_fmt]]),
            columns=('chunk_num', 'chunk_fmt')
        ).to_pickle(f'{feat_base}/train-meta')


if __name__ == '__main__':
    # build_vocabulary()
    extract_training_feat(update_file=False)

    # for doc in list_test_doc():
    #     for x, y in doc:
    #         print(x)
    # x = list_training_doc(True)[0]
    # for y in x:
    #     print(y)
    pass

