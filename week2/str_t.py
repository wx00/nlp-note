
import sys
import re
import numpy as np

from week2 import feat_base, Tag, UNK


def method4():
    str_list = []
    for num in range(loop_count):
        str_list.append(f'{num}')
    out_str = ''.join(str_list)
    return out_str


def method6():
    out_str = ''.join([f'{num}' for num in range(loop_count)])
    return out_str


def method7():
    out_str = ''.join(f'{num}' for num in range(loop_count))
    return out_str


loop_count = 800000

print(sys.version)


# print(  'method4=', timeit.timeit(method4, number=10))
# print(  'method6=', timeit.timeit(method6, number=10))
# print(  'method7=', timeit.timeit(method7, number=10))

def m1():
    x = re.compile('\s')
    for i in '三 月　。 \t\n':
        x.match(i)


def m2():
    for i in '三 月　。 \t\n':
        y = i in (' ', '　', '\t', '\n')


# print('m1=', timeit.timeit(m1, number=1000000))
# print('m2=', timeit.timeit(m2, number=1000000))

# x = re.compile('\s+')
# for i in '三 月　。 \t\n':
#     print(x.match(i))
# print(x.match('三月　十日　（　星期四　）　上午　十時　。'))
# print(x.match(' '))


print(ord('好'), chr(ord('好')))


def some():

    # bi_state_count = np.load(f'{feat_base}/state_count.npy')
    # bi_output_count = np.load(f'{feat_base}/output_count.npy')

    # vocabulary = np.load(f'{feat_base}/vocabulary.npy')
    # state_num, output_num = len(Tag) - 1, len(vocabulary)
    # x = bi_output_count[2]
    # print(x.max() - x.sum())
    # k = 10000000000
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     laplace_numerator = x + k
    #     laplace_denominator = x.sum() + (output_num * k)
    #     y = laplace_numerator / laplace_denominator
    #     z = 1. - y.sum()
    # print(y, x.sum(), y.sum(), z)

    # state_trans_mat = np.load(f'{feat_base}/state_trans.npy')
    # output_prob_mat = np.load(f'{feat_base}/output_prob.npy')
    # print(state_trans_mat.shape)
    # print(output_prob_mat.shape)
    # print(state_trans_mat.sum(axis=1))
    # print(output_prob_mat[:, -1])
    pass
