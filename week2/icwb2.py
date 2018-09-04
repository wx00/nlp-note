import os
import re

import numpy

from week2 import raw_base, feat_base, Tag


def list_file(prof):
    f = f'_{prof}.utf8'
    d = raw_base + f'{prof}/'
    return [d + i + f for i in ['as', 'cityu', 'msr', 'pku']]


def list_doc(prof):
    blank = re.compile('\s')

    def tagging(text):
        buff, words, tags = [], [], []
        for w in text:
            if not blank.match(w):
                buff.append(w)
            elif buff:
                if len(buff) == 1:
                    words.append(buff[0])
                    tags.append(Tag.FIN.value)
                else:
                    words.extend(buff)
                    tags.append(Tag.BEG.value)
                    tags.extend(Tag.MID.value for _ in range(len(buff) - 2))
                    tags.append(Tag.END.value)
                buff.clear()
        return words, tags

    def document(file):
        with open(file, encoding='utf-8') as ff:
            line = ff.readline()
            while line:
                yield tagging(line)
                line = ff.readline()

    return [document(f) for f in list_file(prof)]


# docs = list_doc('training')
# doc = docs[0]
# for i in doc:
#     print(i)
#     break


