import os
import re
import bisect
import numpy as np

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
    for doc in documents:
        for line in doc:
            for token in line:
                for c in token:
                    o = ord(c)
                    if o not in word_set:
                        word_set.add(o)
                        bisect.insort(vocabulary, o)

    vocabulary = np.array(vocabulary)
    np.save(f'{feat_base}/vocabulary.npy', vocabulary)


def list_training_doc(tag=False):
    blank = re.compile('\s')

    def tagging(text):
        buff, words, tags = [], [], []
        for w in text:
            if not blank.match(w):
                buff.append(w)
            elif buff:
                words.extend(buff)
                num = len(buff)
                buff.clear()
                if not tag:
                    continue
                if num == 1:
                    tags.append(Tag.FIN.value)
                else:
                    tags.append(Tag.BEG.value)
                    tags.extend(Tag.MID.value for _ in range(num - 2))
                    tags.append(Tag.END.value)
        return words, tags if tag else words

    def document(file):
        with open(file, encoding='utf-8') as ff:
            line = ff.readline()
            while line:
                yield tagging(line)
                line = ff.readline()

    f = '{0}_training.utf8'
    fmt = f'{raw_base}/training/{f}'
    files = [fmt.format(i) for i in ['as', 'cityu', 'msr', 'pku']]
    return [document(f) for f in files]


def list_test_doc():
    pass
    # [f for f in list_file('testing', 'test')]
    # [f for f in list_file('gold', 'test_gold')]


if __name__ == '__main__':
    x = list_training_doc()[0]
    for y in x:
        print(y)
    pass

