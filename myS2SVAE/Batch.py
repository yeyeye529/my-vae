import torch
from torch.autograd import Variable
import random
import numpy as np

class Batch(object):
    def __init__(self, src, tgt, src_lengths, tgt_lengths, batch_size, is_shuffle = False,
                 src_cls = None, tgt_cls = None, use_cls = False):
        '''
        A little class for batch management
        :param src: src list
        :param tgt: tgt list
        :param batch_size: (int) batch size
        Return:
        Torch
        '''
        self.p_src_list = np.array(src)
        self.p_tgt_list = np.array(tgt)
        self.p_src_lengths = np.array(src_lengths)
        self.p_tgt_lengths = np.array(tgt_lengths)
        self.now_begin = -batch_size
        self.now_end = 0
        self.tot_sent_num = len(src)
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.is_add_unparallel = False
        self.use_cls = use_cls
        if tgt_cls is not None and src_cls is not None:
            self.src_cls = np.array(src_cls)
            self.tgt_cls = np.array(tgt_cls)


    def add_unparallel(self, src, tgt, src_lengths, tgt_lengths, mix_rate):
        self.u_src_list = np.array(src)
        self.u_tgt_list = np.array(tgt)
        self.u_src_lengths = np.array(src_lengths)
        self.u_tgt_lengths = np.array(tgt_lengths)
        self.u_mix_rate = mix_rate
        self.u_sample_num = len(src)
        self.is_add_unparallel = True


    def init_batch(self, sample_unparallel = False):
        self.now_begin = -random.randint(0, self.batch_size - 1) + self.batch_size
        self.now_end = self.now_begin + self.batch_size

        if self.is_add_unparallel and sample_unparallel:
            sample_unparallel_num = int(len(self.p_src_list) * self.u_mix_rate)
            print('sample_unparallel', sample_unparallel_num)
            # randp_unparallel = random.randint(0, 10, size=(3, 4))
            randp_unparallel = np.random.permutation(self.u_sample_num)
            self.src_list = np.concatenate((self.p_src_list, self.u_src_list[randp_unparallel, :][:sample_unparallel_num]), axis=0)
            self.tgt_list = np.concatenate((self.p_tgt_list, self.u_tgt_list[randp_unparallel, :][:sample_unparallel_num]), axis=0)
            self.src_lengths = np.concatenate((self.p_src_lengths, self.u_src_lengths[randp_unparallel][:sample_unparallel_num]), axis=0)
            self.tgt_lengths = np.concatenate((self.p_tgt_lengths, self.u_tgt_lengths[randp_unparallel][:sample_unparallel_num]), axis=0)
        else:
            self.src_list = self.p_src_list
            self.tgt_list = self.p_tgt_list
            self.src_lengths = self.p_src_lengths
            self.tgt_lengths = self.p_tgt_lengths

        if (self.is_shuffle):  # Shuffle
            r = np.random.permutation(len(self.src_lengths))
            self.src_list = self.src_list[r, :]
            self.tgt_list = self.tgt_list[r, :]
            self.src_lengths = self.src_lengths[r]
            self.tgt_lengths = self.tgt_lengths[r]
            if self.use_cls:
                self.src_cls = self.src_cls[r]
                self.tgt_cls = self.tgt_cls[r]

    def have_next_batch(self):
        return self.now_end + self.batch_size <= self.tot_sent_num

    def next_batch(self, fix_batch=False):
        if not fix_batch:
            self.now_begin += self.batch_size
            self.now_end += self.batch_size
        # print(self.now_begin, self.now_end)
        p = np.argsort(-self.src_lengths[self.now_begin: self.now_end])
        # torch.LongTensor(self.src_list[self.now_begin: self.now_end][p, :])
        # x = torch.LongTensor(self.tgt_list[self.now_begin: self.now_end][p, :])
        if self.use_cls:
            return Variable(torch.LongTensor(self.src_list[self.now_begin: self.now_end][p, :])).cuda(), \
                   Variable(torch.LongTensor(self.tgt_list[self.now_begin: self.now_end][p, :])).cuda(), \
                   self.src_lengths[self.now_begin: self.now_end][p], \
                   self.tgt_lengths[self.now_begin: self.now_end][p], \
                   self.src_cls[self.now_begin: self.now_end][p], \
                   self.tgt_cls[self.now_begin: self.now_end][p],
        else:
            return Variable(torch.LongTensor(self.src_list[self.now_begin: self.now_end][p, :])).cuda(), \
                   Variable(torch.LongTensor(self.tgt_list[self.now_begin: self.now_end][p, :])).cuda(), \
                   self.src_lengths[self.now_begin: self.now_end][p], \
                   self.tgt_lengths[self.now_begin: self.now_end][p]
