# -*- coding: utf-8 -*-

"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []


    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `word_prob`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, word_prob):
        """Advance the beam."""

        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if (self.nextYs[-1].cpu().numpy() == self.eos).all():
        # if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self, k=1):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        # print(scores, ids)
        return scores[:k], ids[:k]
        return scores[0], ids[0]    # Why 1?

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]

    def get_hyp_path(self, k):
        path = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            k = self.prevKs[j][k]
            path.append(k)

        return path[::-1]




if __name__ == '__main__':

    vocab = {'<s>':0, '</s>':1, '<pad>':2, 'a': 3, 'b': 4, 'c': 5}
    word_prob_init = torch.FloatTensor([[0.0, 0.0, 0.0, 0.6, 0.4, 0.0]])
    word_prob = torch.FloatTensor([[0.0, 0.0, 0.0, 0.5, 0.2, 0.3], [0.0, 0.0, 0.9, 0.1, 0.0, 0.0]])
    word_prob2 = torch.FloatTensor([[0.0, 0.3, 0.2, 0.1, 0.1, 0.3], [0.0, 0.6, 0.0, 0.2, 0.3, 0.0]])
    b = Beam(2, vocab)

    print('Step 1:')
    print(b.advance(word_prob_init))
    print(b.get_current_state())
    print(b.get_current_origin())
    print('Step 2:')
    print(b.advance(word_prob))  # Search一步
    print(b.get_current_state())
    print(b.get_current_origin())
    print('Step 3:')
    print(b.advance(word_prob2))  # Search一步
    print(b.get_current_state())
    print(b.get_current_origin())
    print("Get current state:")  # 得到当前最好的两个词的ID, 给下一步当输入用
    print(b.get_current_state())
    # print("Get current origin:")
    # print(b.get_current_origin())
    print('Sort best', b.sort_best())
    print()
    print('Get bset', b.get_best())
    print('Get hyp')
    print(b.get_hyp(0))  # 输出路径：实际的句子
    print(b.get_hyp(1))  # 输出路径：实际的句子
    # print(b.get_hyp(2))  # 输出路径：实际的句子
    print(b.get_best(2))
    # print('Get hyp path')
    # print(b.get_hyp_path(0))




