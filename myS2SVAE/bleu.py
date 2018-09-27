# coding=utf-8
# Copyright 2017 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import Counter


def closest_length(candidate, reference):
    clen = candidate
    closest_diff = 9999
    closest_len = 9999

    # for reference in references:
    rlen = reference
    diff = abs(rlen - clen)

    if diff < closest_diff:
        closest_diff = diff
        closest_len = rlen
    elif diff == closest_diff:
        closest_len = rlen if rlen < closest_len else closest_len

    return closest_len


def shortest_length(references):
    return references
    # return min([len(ref) for ref in references])


def modified_precision(candidate, reference, n):
    tngrams = len(candidate) + 1 - n
    counts = Counter([tuple(candidate[i:i + n]) for i in range(tngrams)])

    if len(counts) == 0:
        return 0, 0

    max_counts = {}
    # for reference in references:
    rngrams = len(reference) + 1 - n
    ngrams = [tuple(reference[i:i + n]) for i in range(rngrams)]
    ref_counts = Counter(ngrams)
    for ngram in counts:
        mcount = 0 if ngram not in max_counts else max_counts[ngram]
        rcount = 0 if ngram not in ref_counts else ref_counts[ngram]
        max_counts[ngram] = max(mcount, rcount)

    clipped_counts = {}

    for ngram, count in counts.items():
        clipped_counts[ngram] = min(count, max_counts[ngram])

    return float(sum(clipped_counts.values())), float(sum(counts.values()))


def brevity_penalty(trans_length, refs_length, mode="closest"):
    bp_c = 0.0
    bp_r = 0.0

    for candidate, references in zip(trans_length, refs_length):
        bp_c += candidate

        if mode == "shortest":
            bp_r += shortest_length(references)
        else:
            bp_r += closest_length(candidate, references)

    # Prevent zero divide
    bp_c = bp_c or 1.0

    return math.exp(min(0, 1.0 - bp_r / bp_c))


def bleu(trans, refs, trans_length, refs_length, bp="closest", smooth=True, n=4, weights=None):
    batch_num = len(trans)
    p_norm = [0] * n
    p_denorm = [0] * n

    # for j in range(batch_num):
    #     candidate = trans[j][:trans_length[j]]
    #     references = refs[j]
    #     for i in range(n):
    #         ccount, tcount = modified_precision(candidate, references, i + 1)
    #         p_norm[i] += ccount
    #         p_denorm[i] += tcount

    for j in range(batch_num):
        candidate = trans[j][:trans_length[j]]
        reference = refs[j][:refs_length[j]]
        for i in range(n):
            ccount, tcount = modified_precision(candidate, reference, i + 1)
            p_norm[i] += ccount
            p_denorm[i] += tcount

    bleu_n = [0 for _ in range(n)]

    for i in range(n):
        # Add-one smoothing
        if smooth and i > 0:
            p_norm[i] += 1
            p_denorm[i] += 1
        if p_norm[i] == 0 or p_denorm[i] == 0:
            bleu_n[i] = -9999
        else:
            bleu_n[i] = math.log(float(p_norm[i]) / float(p_denorm[i]))

    if weights:
        if len(weights) != n:
            raise ValueError("len(weights) != n: invalid weight number")
        log_precision = sum([bleu_n[i] * weights[i] for i in range(n)])
    else:
        log_precision = sum(bleu_n) / float(n)

    bp = brevity_penalty(trans_length, refs_length, bp)
    score = bp * math.exp(log_precision)

    return score


if __name__ == '__main__':
    trans = [[0, 19618, 5618, 13342, 2324, 8984, 7744, 9, 8, 1, 1]]
    refs = [[0, 19618, 5618, 13342, 2324, 8984, 1, 3, 3, 3], [0, 19618, 9, 8, 5618, 13342,8,4,2,7,6,1]] # you don't need to care padding!! because bleu = precision?
    # refs = [[[1,2,3]]]
    print(bleu(trans, refs, [10], [8,12]))
