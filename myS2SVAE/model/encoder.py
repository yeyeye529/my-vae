# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding_model = None, n_layers=1, use_cuda = True, use_bidirectional = False,
                 batch_first = False, padding_idx = 1,drop_out = 0.0, rnn_style = 'gru'):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.direction_num = 1 + int(use_bidirectional)
        self.batch_first = batch_first
        self.embedding = embedding_model
        self.dropout = nn.Dropout(drop_out)
        self.rnn_style = rnn_style

        if (self.embedding == None):
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        if rnn_style == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=batch_first,
                              bidirectional=use_bidirectional,
                              dropout=drop_out)
        elif rnn_style == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=batch_first,
                               bidirectional=use_bidirectional,
                               dropout=drop_out)

    def forward(self, input_idx, hidden, sent_length = None, sort = False, input_is_embedding = False):
        '''
        (Assume batch first.)
        :param input_idx: Input word index sequences. Size = (batch_size x max_sent_len)
        :param hidden: Initial hidden state. Size = ((direction_number + layer) x batch_size x hidden_size)
        :param sent_length: Sentence lengths. List, size = (batch_size)
        :param sort: Whether the input_idx and sent_length are sorted by descending sentence length.
        :return:
        output_unpack: hidden states in all time steps (last layer). Size = (batch_size x src_len x (hidden_size x direction_num))
        hidden: hidden state of the last time step. Size = (batch_size x (direction_num + layer) x hidden_size)
        '''
        if not isinstance(sent_length, np.ndarray):
            sent_length = np.array(sent_length)

        if input_is_embedding:
            embedded_dropout = input_idx
        else:
            embedded = self.embedding(input_idx)
            embedded_dropout = self.dropout(embedded)
        if (self.batch_first == False):
            embedded_dropout = torch.transpose(embedded_dropout, 0, 1) # seq_len * batch * input_size

        if sort == False or input_is_embedding:
            p = np.argsort(-sent_length)
            p_inverse = [0] * len(p)
            for i, v in enumerate(p):
                p_inverse[v] = i
            embedded_dropout_sort = embedded_dropout[p, :]
            sent_length_sort = sent_length[p]
        else:
            embedded_dropout_sort = embedded_dropout
            sent_length_sort = sent_length

        embedded_padded = torch.nn.utils.rnn.pack_padded_sequence(embedded_dropout_sort, sent_length_sort, batch_first=self.batch_first)
        # h_0 :  (num_layers * num_directions, batch, hidden_size)
        # input: (seq_len, batch, input_size)
        output, hidden = self.rnn(embedded_padded, hidden)
        # output: (seq_len x batch x hidden_size*num_direction) h_t from the last layer of RNN
        # h_n: (num_layers*num_directions, batch, hidden_size) hidden state for t=seq_len
        output_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                               batch_first = self.batch_first)


        if sort == False:
            output_unpack = output_unpack[p_inverse,:]

        if self.rnn_style == 'lstm':
            hidden = hidden[0]  # (h_n, c_n)
        if self.batch_first == True:
            hidden = hidden.transpose(0, 1)

        return output_unpack, hidden

    def initHidden(self, batch_size):
        if self.rnn_style == 'lstm':
            h_0 = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
            if self.use_cuda:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()
            result = (h_0, c_0)
        elif self.rnn_style == 'gru':
            result = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
            if self.use_cuda:
                result = result.cuda()
        return result
