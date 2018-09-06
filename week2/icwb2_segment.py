import numpy as np
import pandas as pd

from week2 import feat_base, Tag, UNK
from week2.icwb2 import list_test_doc, traverse_doc


def load_training_sequence():

    meta = pd.read_pickle(f'{feat_base}/train-meta').iloc[0]
    chunk_num, chunk_fmt = int(meta['chunk_num']), meta['chunk_fmt']

    chunk_name = f'{feat_base}/{chunk_fmt}'
    chunk_files = [chunk_name.format(i) for i in range(chunk_num)]
    for chunk in chunk_files:
        yield np.load(chunk)


def train(smooth_k=None):
    vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    state_num, output_num = len(Tag) - 1, len(vocabulary)

    if smooth_k is None:
        smooth_k = 1
    elif isinstance(smooth_k, np.ndarray):
        smooth_k = smooth_k.reshape((4, 1))

    # index -1 is state0 (Tag.SEP)
    bi_state_count = np.zeros((state_num+1, state_num), dtype=np.int64)
    bi_output_count = np.zeros((state_num, output_num), dtype=np.int32)

    print('training start')
    for seq in load_training_sequence():
        _, n = seq.shape
        x, y = seq

        prev_word, prev_tag = None, None
        for i in range(n):
            word, tag = x[i], y[i]
            if prev_tag is not None and tag != Tag.SEP.value:
                bi_state_count[prev_tag, tag] += 1
                bi_output_count[tag, word] += 1
            prev_tag = tag
        del x, y, seq
    print('training finish')

    np.save(f'{feat_base}/state_count.npy', bi_state_count)
    np.save(f'{feat_base}/output_count.npy', bi_output_count)

    with np.errstate(divide='ignore', invalid='ignore'):
        state_transfer_mat = np.nan_to_num(
            bi_state_count / bi_state_count.sum(axis=1).reshape(-1, 1))

        laplace_numerator = bi_output_count + smooth_k
        laplace_denominator = bi_output_count.sum(axis=1).reshape(-1, 1) + (output_num * smooth_k)
        output_probability_mat = np.nan_to_num(laplace_numerator / laplace_denominator)

        unknown_output_prob = 1. - output_probability_mat.sum(axis=1).reshape(-1, 1)
        output_probability_mat = np.append(output_probability_mat, unknown_output_prob, axis=1)

        print(unknown_output_prob)

    np.save(f'{feat_base}/state_trans.npy', state_transfer_mat)
    np.save(f'{feat_base}/output_prob.npy', output_probability_mat)


def viterbi_search(seq, state_prob_mat, output_prob_mat):
    t = len(seq)
    assert t > 1

    state_trace = []
    prob_trace = np.zeros((t, 4))

    prev_state = [Tag.SEP.value] * 4
    for i in range(t):
        output_prob = output_prob_mat[:, seq[i]]
        state_prob = state_prob_mat[prev_state] * output_prob
        max_prob_state = np.argmax(state_prob, axis=1).tolist()
        max_prob = state_prob[range(4), max_prob_state]

        if i == 0:
            prob_trace[0, :] = max_prob
        else:
            prob_trace[i, :] = prob_trace[i-1, :] * max_prob

        state_trace.append(max_prob_state)
        prev_state = max_prob_state

    i = np.argmax(prob_trace[-1])
    return list(zip(*state_trace))[i]


def validate():

    state_trans_mat = np.load(f'{feat_base}/state_trans.npy')
    output_prob_mat = np.load(f'{feat_base}/output_prob.npy')

    vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    word_set = set(vocabulary.tolist())

    def ch_to_int(ch):
        o = ord(ch)
        return UNK if o not in word_set else np.searchsorted(vocabulary, o)

    def decode(text, tags):
        text = text.encode('utf-8').decode('utf-8-sig').strip()
        assert len(text) == len(tags)

        buff, tokens = [], []

        def flush():
            tokens.append(''.join(buff))
            buff.clear()

        for i in range(len(tags)):
            buff.append(text[i])
            if tags[i] in (Tag.FIN.value, Tag.END.value):
                flush()

        if len(buff):
            flush()

        return tokens

    def shrink(tags):
        idx, mixture = None, []
        for i in range(len(tags)):
            if tags[i] in (Tag.FIN.value, Tag.END.value):
                mixture.append((i if idx is None else idx << 32) | i)
                # pos.append((i if idx is None else idx, i))
                idx = None
            elif idx is None:
                idx = i
        if idx is not None:
            mixture.append((idx << 32) | (len(tags) - 1))
            # pos.append((idx, len(tags)-1))
        return set(mixture)
        # return pos

    for doc in list_test_doc():
        for x, y in doc:
            sample = traverse_doc(x, [], None, converter=ch_to_int)
            target = traverse_doc(y, None, [])
            if len(sample) < 2:
                continue

            guess = viterbi_search(sample, state_trans_mat, output_prob_mat)
            predict, expect = shrink(guess), shrink(target)

            diff = predict.difference(expect)
            err_num, right_num = len(diff), len(predict) - len(diff)
            # if len(diff) > 5:
            #     print(decode(x, guess))
            #     print(y)


if __name__ == '__main__':
    # train(smooth_k=np.array([1, 1, 10000000000, 1]))
    validate()
    pass





