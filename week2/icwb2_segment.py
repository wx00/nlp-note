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


def train(smooth_output=None, smooth_state=None):
    vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    state_num, output_num = len(Tag) - 1, len(vocabulary)

    if isinstance(smooth_output, np.ndarray):
        smooth_output = smooth_output.reshape((4, 1))
    if isinstance(smooth_state, np.ndarray):
        smooth_state = smooth_state.reshape((4, 1))

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

    def laplace_smoothing(numerator, denominator, v, k):
        laplace_numerator = numerator + k
        laplace_denominator = denominator + (v * k)
        return np.nan_to_num(laplace_numerator / laplace_denominator)

    with np.errstate(divide='ignore', invalid='ignore'):
        if smooth_state is not None:
            state_transfer_mat = laplace_smoothing(
                bi_state_count, bi_state_count.sum(axis=1).reshape(-1, 1),
                state_num, smooth_state)
        else:
            state_transfer_mat = np.nan_to_num(
                bi_state_count / bi_state_count.sum(axis=1).reshape(-1, 1))

        if smooth_output is not None:
            output_emit_mat = laplace_smoothing(
                bi_output_count, bi_output_count.sum(axis=1).reshape(-1, 1),
                output_num, smooth_output)
        else:
            output_emit_mat = np.nan_to_num(
                bi_output_count / bi_output_count.sum(axis=1).reshape(-1, 1))

        unknown_output_prob = 1. - output_emit_mat.sum(axis=1).reshape(-1, 1)
        output_emit_mat = np.append(output_emit_mat, unknown_output_prob, axis=1)

        print(unknown_output_prob)

    np.save(f'{feat_base}/state_trans.npy', state_transfer_mat)
    np.save(f'{feat_base}/output_emit.npy', output_emit_mat)


def viterbi_search(seq, state0_prob, trans_prob_mat, emit_prob_mat):
    n = len(seq)
    assert n > 1

    state_trace = []
    max_prob_trace = np.zeros((n, 4))
    max_prob_trace[0, :] = state0_prob + emit_prob_mat[:, seq[0]]

    for t in range(1, n):
        emit_prob = emit_prob_mat[:, seq[t]]  # prob([F, B, M, E] -> seq[t])
        prev_prob = max_prob_trace[t-1, :]  # max_prob([F, B, M, E])

        max_prev_state = [Tag.SEP.value] * 4
        for k in range(4):
            trans_prob = trans_prob_mat[:, k]  # prob([F, B, M, E] -> k)
            # max_prob([F, B, M, E]) * prob([F, B, M, E] -> k) * prob(k -> seq[t])
            prob_k = prev_prob + trans_prob + emit_prob[k]
            max_prev_state[k] = np.argmax(prob_k)
            max_prob_trace[t, k] = prob_k[max_prev_state[k]]
        state_trace.append(max_prev_state)

    m = np.argmax(max_prob_trace[-1])
    max_path = [m]
    for i in range(n-2, -1, -1):
        m = state_trace[i][m]
        max_path.append(m)

    max_path.reverse()
    return max_path


def validate():

    state_trans_mat = np.load(f'{feat_base}/state_trans.npy')
    output_emit_mat = np.load(f'{feat_base}/output_emit.npy')
    # print(state_trans_mat.shape)
    # print(output_emit_mat.shape)
    # print(state_trans_mat.sum(axis=1))
    # print(state_trans_mat)
    # print(output_emit_mat.sum(axis=1))
    # print(output_emit_mat)
    # exit(0)

    state_trans_mat = np.log(state_trans_mat)
    output_emit_mat = np.log(output_emit_mat)

    vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    word_set = set(vocabulary.tolist())

    def ch_to_int(ch):
        o = ord(ch)
        return UNK if o not in word_set else np.searchsorted(vocabulary, o)

    def decode(text, tags):
        text = text.encode('utf-8').decode('utf-8-sig').strip()
        assert len(text) == len(tags)

        buff, tokens = [], []
        for i in range(len(tags)):
            buff.append(text[i])
            if tags[i] in (Tag.FIN.value, Tag.END.value):
                tokens.append(''.join(buff))
                buff.clear()

        if len(buff):
            tokens.append(''.join(buff))
            buff.clear()

        return tokens

    def shrink(tags):
        p, mixture = None, []
        for i in range(len(tags)):
            if tags[i] in (Tag.FIN.value, Tag.END.value):
                mixture.append((i if p is None else p << 32) | i)
                p = None
            elif p is None:
                p = i
        if p is not None:
            mixture.append((p << 32) | (len(tags) - 1))
        return set(mixture)

    total, correct, error = 0, 0, 0
    for doc in list_test_doc():
        for x, y in doc:
            sample = traverse_doc(x, [], None, converter=ch_to_int)
            target = traverse_doc(y, None, [])
            if len(sample) < 2:
                continue

            guess = viterbi_search(
                sample, state_trans_mat[-1], state_trans_mat[:-1], output_emit_mat)
            # print(decode(sample, guess))

            predict, expect = shrink(guess), shrink(target)
            diff = predict.difference(expect)
            correct += len(predict) - len(diff)
            error += len(diff)
            total += len(expect)

    recall = correct/total
    precision = correct/(correct + error)

    print(f'Recall: {recall}')
    print(f'Precision: {precision}')


if __name__ == '__main__':
    # train(smooth_output=np.array([1, 1, 10000000000, 1]))
    validate()
    pass





