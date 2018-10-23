# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Attention import DotAttention
from Attention import VaeAttention
from Attention import GeneralAttention
from utils.Utils import sequence_mask
from utils.Utils import sparse_to_matrix
import random
import math

from model.encoder import EncoderRNN

MAX_LENGTH = 128

# class EncoderRNN(nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size, embedding_model = None, n_layers=1, use_cuda = True, use_bidirectional = False,
#                  batch_first = False, padding_idx = 1,drop_out = 0.0, rnn_style = 'gru'):
#         super(EncoderRNN, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.use_cuda = use_cuda and torch.cuda.is_available()
#         self.direction_num = 1 + int(use_bidirectional)
#         self._batch_first = batch_first
#         self.embedding = embedding_model
#         self.dropout = nn.Dropout(drop_out)
#         self.rnn_style = rnn_style
#
#         if (self.embedding == None):
#             self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
#         if rnn_style == 'gru':
#             self.rnn = nn.GRU(input_size=embedding_size,
#                               hidden_size=hidden_size,
#                               num_layers=n_layers, batch_first=batch_first,
#                               bidirectional=use_bidirectional,
#                               dropout=drop_out)
#         elif rnn_style == 'lstm':
#             self.rnn = nn.LSTM(input_size=embedding_size,
#                                hidden_size=hidden_size,
#                                num_layers=n_layers, batch_first=batch_first,
#                                bidirectional=use_bidirectional,
#                                dropout=drop_out)
#
#     def forward(self, input_idx, hidden, sent_length = None, sort = False):
#         '''
#         (Assume batch first.)
#         :param input_idx: Input word index sequences. Size = (batch_size x max_sent_len)
#         :param hidden: Initial hidden state. Size = ((direction_number + layer) x batch_size x hidden_size)
#         :param sent_length: Sentence lengths. List, size = (batch_size)
#         :param sort: Whether the input_idx and sent_length are sorted by descending sentence length.
#         :return:
#         output_unpack: hidden states in all time steps (last layer). Size = (batch_size x src_len x (hidden_size x direction_num))
#         hidden: hidden state of the last time step. Size = (batch_size x (direction_num + layer) x hidden_size)
#         '''
#         if not isinstance(sent_length, np.ndarray):
#             sent_length = np.array(sent_length)
#
#         embedded = self.embedding(input_idx)
#         embedded_dropout = self.dropout(embedded)
#         if (self._batch_first == False):
#             embedded_dropout = torch.transpose(embedded_dropout, 0, 1) # seq_len * batch * input_size
#
#         if sort == False:
#             p = np.argsort(-sent_length)
#             p_inverse = [0] * len(p)
#             for i, v in enumerate(p):
#                 p_inverse[v] = i
#             embedded_dropout_sort = embedded_dropout[p, :]
#             sent_length_sort = sent_length[p]
#         else:
#             embedded_dropout_sort = embedded_dropout
#             sent_length_sort = sent_length
#
#         if np.array(sent_length).all() != None:
#             embedded_padded = torch.nn.utils.rnn.pack_padded_sequence(embedded_dropout_sort, sent_length_sort, batch_first=self._batch_first)
#             # h_0 :  (num_layers * num_directions, batch, hidden_size)
#             # input: (seq_len, batch, input_size)
#             output, hidden = self.rnn(embedded_padded, hidden)
#             # output: (seq_len x batch x hidden_size*num_direction) h_t from the last layer of RNN
#             # h_n: (num_layers*num_directions, batch, hidden_size) hidden state for t=seq_len
#             output_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(output,
#                                                                    batch_first = self._batch_first)
#         else:
#             output, hidden = self.rnn(embedded_dropout, hidden)
#             output_unpack = output
#
#         if sort == False:
#             output_unpack = output_unpack[p_inverse,:]
#
#         if self.rnn_style == 'lstm':
#             hidden = hidden[0]  # (h_n, c_n)
#         if self._batch_first == True:
#             hidden = hidden.transpose(0, 1)
#
#         return output_unpack, hidden
#
#     def initHidden(self, batch_size):
#         if self.rnn_style == 'lstm':
#             h_0 = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
#             c_0 = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
#             if self.use_cuda:
#                 h_0 = h_0.cuda()
#                 c_0 = c_0.cuda()
#             result = (h_0, c_0)
#         elif self.rnn_style == 'gru':
#             result = Variable(torch.zeros(self.n_layers * self.direction_num, batch_size, self.hidden_size))
#             if self.use_cuda:
#                 result = result.cuda()
#         return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, encoder_layer, attn_obj = None, embedding_model = None, n_layers=1, dropout_p=0.0, max_length=MAX_LENGTH,
                 use_cuda = True, batch_first = False, padding_idx = 1, context_gate = None, use_bidiretional_encoder = False, rnn_style = 'gru',
                 use_output_layer = False, use_first_encoder_hidden_state = False, output_bow_loss = False):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.n_layers = n_layers
        self.encoder_layer = encoder_layer
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.embedding = embedding_model
        self.use_bidiretional_encoder = use_bidiretional_encoder
        self.rnn_style = rnn_style
        self.attention = attn_obj   # Attention class
        self.output_bow_loss = output_bow_loss

        self.use_first_encoder_hidden_state = use_first_encoder_hidden_state  # == False: use last hidden state
        if self.use_first_encoder_hidden_state:
            self.encoder2decoder = nn.Linear(
                self.hidden_size,
                self.hidden_size
            )
        else:
            self.encoder2decoder = nn.Linear(
                self.hidden_size * (int(self.use_bidiretional_encoder) + 1),
                self.hidden_size
            )

        if (self.embedding == None):
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_style == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size + self.attention.attn_output_dim,
                              hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=False,
                              dropout=dropout_p)
        elif rnn_style == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size + self.attention.attn_output_dim,
                               hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=False,
                               dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        print('use cuda: ', self.use_cuda)

        self.use_output_layer = use_output_layer
        if self.use_output_layer:
            self.l_size = hidden_size
            self.Uo = nn.Linear(self.hidden_size, 2 * self.l_size, bias=False)
            self.Vo = nn.Linear(self.embedding_size, 2 * self.l_size, bias=False)
            self.Co = nn.Linear(self.hidden_size * 2, 2 * self.l_size, bias=False)


    def forward(self, input_idx, contexts, context_lengths, encoder_ht, padding = True, use_z_embedding = False, z_embedding = None):
        # Notice: sequence First!!
        # Translation step: input_idx only have one step.
        '''
        :param input_idx: Input word index sequences.
        :param contexts: Hidden states which is output by encoder.
        :param context_lengths: Sentence length of source sentences.
        :param encoder_ht: Last hidden state which is output by encoder.
        :param padding: Padding output vectors or not.
        :param use_z_embedding: Whether use z_embedding vector as the first embedding input of decoder.
        :param z_embedding: .
        :return: output_log_softmax, hiddens, attns
        output_log_softmax: Output vector: log probabilities of predicted word of every step.
        hiddens: all hidden states.
        attns: attns['scores'], attns['vectors']: attention vectors and scores.
        '''
        if self._batch_first:
            step_time = input_idx.size(1)
        else:
            step_time = input_idx.size(0)
        if step_time == 1:
            one_step = True
        else:
            one_step = False

        embedded = self.embedding(input_idx)
        embedded_dropout = self.dropout(embedded)
        if self.use_bidiretional_encoder:
            if encoder_ht.size(1) != 1:
                if self.use_first_encoder_hidden_state:  # Use the first time step leftward hidden state as the initial state.
                    encoder_ht = contexts[:, 0, self.hidden_size:].unsqueeze(1)
                else:
                    encoder_ht = torch.chunk(encoder_ht, self.encoder_layer, dim=1)[-1]   # Only consider the last layer
                    encoder_ht = encoder_ht.contiguous().view(encoder_ht.size(0), 1, -1)
                hidden = torch.tanh(self.encoder2decoder(encoder_ht))
            else:
                hidden = encoder_ht
            if self.attention.attn_type == 'dot':
                contexts = torch.tanh(self.encoder2decoder(contexts))  # If use dot attention, need to transform encoder hidden to decoder hidden space
        else:
            hidden = encoder_ht

        rnn_outputs_collect = []
        hidden_collect = []
        attn_scores_collect = []
        attn_outputs_collect = []
        attns = {'scores': None, 'vectors': None}

        step_time = max(step_time, 2)  # In case only has one step

        if self._batch_first == True:   # Must be (seq_len x batch_size x *)
            hidden = hidden.transpose(0, 1)
            contexts = contexts.transpose(0, 1)
            embedded_dropout = embedded_dropout.transpose(0, 1)
            if use_z_embedding:
                z_embedding = z_embedding.transpose(0, 1)  # 1 x 1 x *

        if self.rnn_style == 'lstm':
            h_t = hidden
            c_t = hidden

        if use_z_embedding:  # If you use z_embedding as the first word embedding of decoder.
            attn_outputs, attn_scores = self.attention(
                hidden.transpose(0, 1).contiguous(),  # (batch, 1, d)
                contexts.transpose(0, 1),  # (batch, contxt_len, d)
                context_lengths=context_lengths
            )
            rnn_input = torch.cat([attn_outputs, z_embedding], 2)
            if self.rnn_style == 'lstm':
                _, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))
                hidden = h_t
            elif self.rnn_style == 'gru':
                _, hidden = self.rnn(rnn_input, hidden)


        for step in range(step_time - 1): # -1 because we don't care <eos> decoding what
            attn_outputs, attn_scores = self.attention(
                hidden.transpose(0, 1).contiguous(),  # (batch, 1, d)
                contexts.transpose(0, 1),  # (batch, context_len, d)
                context_lengths=context_lengths
            )
            rnn_input = torch.cat([attn_outputs, embedded_dropout[step].unsqueeze(0)], 2)

            if self.use_output_layer:
                rnn_output = self.Uo(hidden) + self.Vo(embedded_dropout[step].unsqueeze(0)) + self.Co(attn_outputs)
                rnn_output = torch.nn.functional.max_pool1d(rnn_output, 2)

            if self.rnn_style == 'lstm':
                _, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))
                hidden = h_t
            elif self.rnn_style == 'gru':
                _, hidden = self.rnn(rnn_input, hidden)

            if not self.use_output_layer:
                rnn_output = hidden


            rnn_output = self.dropout(rnn_output)  # (input_len, batch, d)

            rnn_outputs_collect.append(rnn_output)
            hidden_collect.append(hidden)
            attn_scores_collect.append(attn_scores)
            attn_outputs_collect.append(attn_outputs)

        # if self.context_gate is not None:
        #     outputs = self.context_gate(
        #         emb.view(-1, emb.size(2)),
        #         rnn_output.view(-1, rnn_output.size(2)),
        #         attn_outputs.view(-1, attn_outputs.size(2))
        #     )
        #     outputs = outputs.view(input_len, input_batch, self.hidden_size)
        #     outputs = self.dropout(outputs)
        # else:
        #     outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        rnn_outputs = torch.cat(rnn_outputs_collect, 0)
        hiddens = torch.cat(hidden_collect, 0)
        attns['scores'] = torch.cat(attn_scores_collect, 0)
        attns['vectors'] = torch.cat(attn_outputs_collect, 0)
        nn_out = self.out(rnn_outputs.view(-1, rnn_outputs.size(2))).view(rnn_outputs.size(0), rnn_outputs.size(1), -1)
        # output_log_softmax = F.log_softmax(self.out(rnn_outputs.view(-1, rnn_outputs.size(2))), dim=1)  # ???
        output_log_softmax = F.log_softmax(nn_out, dim=2)
        # print(output_log_softmax.size())
        # output_log_softmax = output_log_softmax.view(rnn_outputs.size(0), rnn_outputs.size(1), -1)

        # output_log_sofmax: str_len x batch_size x vocab_size
        # hiddens: str_len x batch_size x hidden_size
        # Padding
        if one_step == False and padding == True:
            output_log_softmax = F.pad(output_log_softmax, (0, 0, 0, 0, 1, 0))
            hiddens = F.pad(hiddens, (0, 0, 0, 0, 1, 0))
            nn_out = F.pad(nn_out, (0, 0, 0, 0, 1, 0))
            attns['scores'] = F.pad(attns['scores'], (0, 0, 0, 0, 1, 0))
            attns['vectors'] = F.pad(attns['vectors'], (0, 0, 0, 0, 1, 0))

        if self._batch_first:
            output_log_softmax = output_log_softmax.transpose(0, 1)
            hiddens = hiddens.transpose(0, 1)
            nn_out = nn_out.transpose(0, 1)
            attns['scores'] = attns['scores'].transpose(0, 1)
            attns['vectors'] = attns['vectors'].transpose(0, 1)

        if self.output_bow_loss:
            if self.training:
                # nn_out = None
                return (output_log_softmax, nn_out), hiddens, attns
            else:
                return output_log_softmax, hiddens, attns
        return output_log_softmax, hiddens, attns

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

# Almost the same as AttnDecoderRNN
class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, encoder_layer, embedding_model = None, n_layers=1, dropout_p=0.0, max_length=MAX_LENGTH,
                 use_cuda = True, batch_first = False, padding_idx = 1, context_gate = None, use_bidiretional_encoder = False, rnn_style = 'gru',
                 decoder_input_dropout = 0.0):
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.n_layers = n_layers
        self.encoder_layer = encoder_layer
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.embedding = embedding_model
        self.use_bidiretional_encoder = use_bidiretional_encoder
        self.rnn_style = rnn_style
        self.decoder_input_dropout = decoder_input_dropout
        self.encoder2decoder = nn.Linear(
            self.hidden_size * (int(self.use_bidiretional_encoder) + 1),
            self.hidden_size
        )
        if (self.embedding == None):
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding?
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_style == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=False,
                              dropout=dropout_p)
        elif rnn_style == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=False,
                               dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        print('use cuda: ', self.use_cuda)

    def forward(self, input_idx, encoder_ht, padding = True):
        # print("Decoder without attention")
        # Notice: Squence First
        # Translation step: input_idx only have one step.
        if self._batch_first:  # batch x strlen
            step_time = input_idx.size(1)
        else:  # strlen x batch
            step_time = input_idx.size(0)
        if step_time == 1:
            one_step = True
        else:
            one_step = False

        embedded = self.embedding(input_idx)
        embedded_dropout = self.dropout(embedded)
        if self.use_bidiretional_encoder:
            if encoder_ht.size(1) != 1:
                encoder_ht = torch.chunk(encoder_ht, self.encoder_layer, dim=1)[-1]   # Only consider the last layer
                encoder_ht = encoder_ht.contiguous().view(encoder_ht.size(0), 1, -1)
                hidden = torch.tanh(self.encoder2decoder(encoder_ht))
            else:
                hidden = encoder_ht
        else:
            hidden = encoder_ht

        rnn_outputs_collect = []
        hidden_collect = []

        step_time = max(step_time, 2)  # In case only has one step

        if self._batch_first == True:   # Must be (seq_len x batch_size x *)
            hidden = hidden.transpose(0, 1)
            embedded_dropout = embedded_dropout.transpose(0, 1)

        if self.rnn_style == 'lstm':
            h_t = hidden
            c_t = hidden
        for step in range(step_time - 1): # -1 because we don't care <eos> decoding what
            rnn_input = embedded_dropout[step].unsqueeze(0)
            if self.training and random.random() < self.decoder_input_dropout:
                # rnn_input.contiguous().zero_()
                rnn_input = torch.zeros_like(rnn_input)
            if self.rnn_style == 'lstm':
                rnn_output, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))
                hidden = h_t
            elif self.rnn_style == 'gru':
                rnn_output, hidden = self.rnn(rnn_input, hidden)

            rnn_output = self.dropout(rnn_output)  # (input_len, batch, d)

            rnn_outputs_collect.append(rnn_output)
            hidden_collect.append(hidden)

        # if self.context_gate is not None:
        #     outputs = self.context_gate(
        #         emb.view(-1, emb.size(2)),
        #         rnn_output.view(-1, rnn_output.size(2)),
        #         attn_outputs.view(-1, attn_outputs.size(2))
        #     )
        #     outputs = outputs.view(input_len, input_batch, self.hidden_size)
        #     outputs = self.dropout(outputs)
        # else:
        #     outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        rnn_outputs = torch.cat(rnn_outputs_collect, 0)
        hiddens = torch.cat(hidden_collect, 0)
        output_log_softmax = F.log_softmax(self.out(rnn_outputs.view(-1, rnn_outputs.size(2))), dim=1)   # ???
        output_log_softmax = output_log_softmax.view(rnn_outputs.size(0), rnn_outputs.size(1), -1)

        # output_log_sofmax: str_len x batch_size x vocab_size
        # hiddens: str_len x batch_size x hidden_size
        # Padding
        if one_step == False and padding == True:
            output_log_softmax = F.pad(output_log_softmax, (0, 0, 0, 0, 1, 0))
            hiddens = F.pad(hiddens, (0, 0, 0, 0, 1, 0))

        if self._batch_first:
            output_log_softmax = output_log_softmax.transpose(0, 1)
            hiddens = hiddens.transpose(0, 1)
        return output_log_softmax, hiddens, 0

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


# VAE Decoder: use the information from z
class AttnDecoderRNN_vae(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, z_size, encoder_layer, attn_obj = None, embedding_model = None, n_layers=1, dropout_p=0.0, max_length=MAX_LENGTH,
                 use_cuda = True, batch_first = False, padding_idx = 1, context_gate = None, use_bidiretional_encoder = False, rnn_style = 'gru',
                 decoder_input_dropout = 0.0, use_vae_attention = False, vae_attention = None, vae_first_h = False, decoder_use_c = False,
                 use_attention = True, teacher_forcing_rate = 0.0):
        super(AttnDecoderRNN_vae, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.z_size = z_size
        self.n_layers = n_layers
        self.encoder_layer = encoder_layer
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.embedding = embedding_model
        self.use_bidiretional_encoder = use_bidiretional_encoder
        self.rnn_style = rnn_style
        self.attention = attn_obj   # Attention class
        self.use_vae_attention = use_vae_attention
        self.vae_attention = vae_attention
        self.vae_first_h = vae_first_h
        self.decoder_use_c = decoder_use_c
        self.use_attention = use_attention
        self.teacher_forcing_rate = teacher_forcing_rate

        print("Use attention:", use_attention)

        self.encoder2decoder = nn.Linear(
            self.hidden_size * (int(self.use_bidiretional_encoder) + 1),
            self.hidden_size
        )

        self.relu = nn.ReLU()

        if self.use_attention:
            self.rnn_cell_size = self.z_size + self.embedding_size + self.hidden_size
            if self.use_vae_attention:
                self.rnn_cell_size = self.embedding_size + self.hidden_size
        else:
            self.rnn_cell_size = self.embedding_size

        print('rnn_cell_size', self.rnn_cell_size)
        # self.rnn_input_combine = nn.Linear(self.z_size + self.embedding_size + self.hidden_size, self.embedding_size)
        if use_vae_attention or vae_first_h:
            if self.decoder_use_c:
                self.z2decoder = nn.Linear(self.z_size * 2, self.hidden_size)
            else:
                self.z2decoder = nn.Linear(self.z_size, self.hidden_size)

        self.decoder_input_dropout = decoder_input_dropout
        if (self.embedding == None):
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_style == 'gru':
            self.rnn = nn.GRU(input_size=self.rnn_cell_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=False,
                              dropout=dropout_p)
        elif rnn_style == 'lstm':
            self.rnn = nn.LSTM(input_size=self.rnn_cell_size,
                               hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=False,
                               dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        print('use cuda: ', self.use_cuda)


    def feedback(self):
        pass

    def forward(self, input_idx, contexts, context_lengths, encoder_ht, z, padding = True,
                first_step = True, c = None):
        # Notice: Squence First
        # Translation step: input_idx only have one step.
        if self._batch_first:  # batch x strlen
            step_time = input_idx.size(1)
        else:  # strlen x batch
            step_time = input_idx.size(0)
        if step_time == 1:
            one_step = True
        else:
            one_step = False

        embedded = self.embedding(input_idx)
        embedded_dropout = self.dropout(embedded)  # 在下面一起drop了。
        # embedded_dropout = embedded
        if first_step and (self.use_vae_attention or self.vae_first_h):
            if self.decoder_use_c:
                z_trans = self.z2decoder(torch.cat([c.unsqueeze(1), z], 2))
            else:
                z_trans = self.z2decoder(z)
        else:
            z_trans = z

        if self.use_bidiretional_encoder:
            if encoder_ht.size(1) != 1:
                encoder_ht = torch.chunk(encoder_ht, self.encoder_layer, dim=1)[-1]   # Only consider the last layer!
                encoder_ht = encoder_ht.contiguous().view(encoder_ht.size(0), 1, -1)
                hidden = torch.tanh(self.encoder2decoder(encoder_ht))
            else:
                hidden = encoder_ht
            contexts = torch.tanh(self.encoder2decoder(contexts)) if contexts is not None else contexts
        else:
            hidden = encoder_ht

        if (first_step) and self.vae_first_h:
            # print('hidden:', hidden.size())
            hidden = hidden + torch.tanh(z_trans)
            # print('hidden:', hidden.size())

        rnn_outputs_collect = []
        hidden_collect = []
        attn_scores_collect = []
        attn_outputs_collect = []
        logits_collect = []
        attns = {'scores': None, 'vectors': None}

        step_time = max(step_time, 2)  # In case only has one step

        if self._batch_first == True:   # Must be (seq_len x batch_size x *)
            hidden = hidden.transpose(0, 1)
            contexts = contexts.transpose(0, 1) if contexts is not None else contexts
            z_trans = z_trans.transpose(0, 1)
            embedded_dropout = embedded_dropout.transpose(0, 1)

        if self.rnn_style == 'lstm':
            h_t = hidden
            c_t = hidden

        # teacher forcing:
        use_teacher_forcing = True if random.random() < self.teacher_forcing_rate else False
        for step in range(step_time - 1): # -1 because we don't care <eos> decoding what
            if self.use_attention:
                print("Use attention?")
                attn_outputs, attn_scores = self.attention(
                    hidden.transpose(0, 1).contiguous(),  # (batch, 1, d)
                    contexts.transpose(0, 1),  # (batch, contxt_len, d)
                    context_lengths=context_lengths
                )
                if self.use_vae_attention:
                    attn_outputs, _, _ = self.vae_attention(attn_outputs, z_trans, hidden)

            if not use_teacher_forcing:
                # Drop out some decoder input:
                if self.training and random.random() < self.decoder_input_dropout:
                    input_embedding = Variable(torch.zeros(embedded_dropout[step].unsqueeze(0).size()))
                    if self.use_cuda:
                        input_embedding = input_embedding.cuda()
                else:
                    input_embedding = embedded_dropout[step].unsqueeze(0)
                    # 1 * batch_size * embedding_size
            else:
                if step == 0:
                    input_embedding = embedded_dropout[0].unsqueeze(0)
                else:
                    input_embedding = embedded_dropout

            if z_trans.dim() == 2:
                z_trans = z_trans.unsqueeze(0)

            if self.use_attention:
                if self.use_vae_attention:
                    rnn_input = torch.cat([attn_outputs, input_embedding], 2)
                else:
                    rnn_input = torch.cat([attn_outputs, input_embedding, z_trans], 2)
            else:
                rnn_input = input_embedding

            if self.rnn_style == 'lstm':
                rnn_output, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))
                hidden = h_t
            elif self.rnn_style == 'gru':
                rnn_output, hidden = self.rnn(rnn_input, hidden)

            rnn_output = self.dropout(rnn_output)  # (input_len, batch, d)
            logits = self.out(rnn_output)

            # rnn_outputs_collect.append(rnn_output)
            logits_collect.append(logits)
            hidden_collect.append(hidden)

            if self.use_attention:
                attn_scores_collect.append(attn_scores)
                attn_outputs_collect.append(attn_outputs)

            if use_teacher_forcing:
                topv, topi = logits.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(0)  # detach from history as input
                embedded = self.embedding(decoder_input)
                embedded_dropout = self.dropout(embedded)

        # if self.context_gate is not None:
        #     outputs = self.context_gate(
        #         emb.view(-1, emb.size(2)),
        #         rnn_output.view(-1, rnn_output.size(2)),
        #         attn_outputs.view(-1, attn_outputs.size(2))
        #     )
        #     outputs = outputs.view(input_len, input_batch, self.hidden_size)
        #     outputs = self.dropout(outputs)
        # else:
        #     outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        # rnn_outputs = torch.cat(rnn_outputs_collect, 0)
        logits = torch.cat(logits_collect, 0)
        hiddens = torch.cat(hidden_collect, 0)
        if self.use_attention:
            attns['scores'] = torch.cat(attn_scores_collect, 0)
            attns['vectors'] = torch.cat(attn_outputs_collect, 0)
        output_log_softmax = F.log_softmax(logits.view(-1, logits.size(2)), dim=1)   # ???
        output_log_softmax = output_log_softmax.view(logits.size(0), logits.size(1), -1)

        # output_log_sofmax: str_len x batch_size x vocab_size
        # hiddens: str_len x batch_size x hidden_size
        # Padding
        if one_step == False and padding == True:
            output_log_softmax = F.pad(output_log_softmax, (0, 0, 0, 0, 1, 0))
            hiddens = F.pad(hiddens, (0, 0, 0, 0, 1, 0))
            if self.use_attention:
                attns['scores'] = F.pad(attns['scores'], (0, 0, 0, 0, 1, 0))
                attns['vectors'] = F.pad(attns['vectors'], (0, 0, 0, 0, 1, 0))

        if self._batch_first:
            output_log_softmax = output_log_softmax.transpose(0, 1)
            hiddens = hiddens.transpose(0, 1)
            if self.use_attention:
                attns['scores'] = attns['scores'].transpose(0, 1)
                attns['vectors'] = attns['vectors'].transpose(0, 1)
        return output_log_softmax, hiddens, attns

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        result = result.cuda() if self.use_cuda else result
        return result

# VAE Decoder: without attention
class DecoderRNN_vae(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, z_size, encoder_layer, embedding_model = None, n_layers=1, dropout_p=0.0, max_length=MAX_LENGTH,
                 use_cuda = True, batch_first = False, padding_idx = 1, context_gate = None, use_bidiretional_encoder = False, rnn_style = 'gru',
                 decoder_input_dropout = 0.0):
        super(DecoderRNN_vae, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.z_size = z_size
        self.n_layers = n_layers
        self.encoder_layer = encoder_layer
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.embedding = embedding_model
        self.use_bidiretional_encoder = use_bidiretional_encoder
        self.rnn_style = rnn_style
        self.encoder2decoder = nn.Linear(
            self.hidden_size * (int(self.use_bidiretional_encoder) + 1),
            self.hidden_size
        )
        self.relu = nn.ReLU()
        self.fc_z2z = nn.Linear(self.z_size, self.z_size)
        # self.rnn_input_combine = nn.Linear(self.z_size + self.embedding_size + self.hidden_size, self.embedding_size)

        self.decoder_input_dropout = decoder_input_dropout
        if (self.embedding == None):
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_style == 'gru':
            self.rnn = nn.GRU(input_size=self.z_size + self.embedding_size,  # embedding_size + hidden_size + z_size
                              hidden_size=hidden_size,
                              num_layers=n_layers, batch_first=False,
                              dropout=dropout_p)
        elif rnn_style == 'lstm':
            self.rnn = nn.LSTM(input_size=self.z_size + self.embedding_size,  # embedding_size + hidden_size + z_size
                               hidden_size=hidden_size,
                               num_layers=n_layers, batch_first=False,
                               dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        print('use cuda: ', self.use_cuda)

    def forward(self, input_idx, contexts, context_lengths, encoder_ht, z, padding = True):
        # Notice: Squence First!!
        # Translation step: input_idx only have one step.
        if self._batch_first:  # batch x strlen
            step_time = input_idx.size(1)
        else:  # strlen x batch
            step_time = input_idx.size(0)
        if step_time == 1:
            one_step = True
        else:
            one_step = False

        embedded = self.embedding(input_idx)
        embedded_dropout = self.dropout(embedded)
        # z_trans = self.relu(self.fc_z2z(z))
        z_trans = z

        if self.use_bidiretional_encoder:
            if encoder_ht.size(1) != 1:
                encoder_ht = torch.chunk(encoder_ht, self.encoder_layer, dim=1)[-1]   # Only consider the last layer!
                encoder_ht = encoder_ht.contiguous().view(encoder_ht.size(0), 1, -1)
                hidden = torch.tanh(self.encoder2decoder(encoder_ht))
            else:
                hidden = encoder_ht
            contexts = torch.tanh(self.encoder2decoder(contexts))
        else:
            hidden = encoder_ht

        rnn_outputs_collect = []
        hidden_collect = []

        step_time = max(step_time, 2)  # In case only has one step

        if self._batch_first == True:   # Must be (seq_len x batch_size x *)
            hidden = hidden.transpose(0, 1)
            contexts = contexts.transpose(0, 1)
            z_trans = z_trans.transpose(0, 1)
            embedded_dropout = embedded_dropout.transpose(0, 1)

        if self.rnn_style == 'lstm':
            h_t = hidden
            c_t = hidden
        for step in range(step_time - 1): # -1 because we don't care <eos> decoding what
            # Drop out some decoder input: Actually should use embedding of <unk>
            if self.training and random.random() < self.decoder_input_dropout:
                # input_embedding = self.embedding(input_idx)
                input_embedding = Variable(torch.zeros(embedded_dropout[step].unsqueeze(0).size()))
                if self.use_cuda:
                    input_embedding = input_embedding.cuda()
            else:
                input_embedding = embedded_dropout[step].unsqueeze(0)

            if z_trans.dim() == 2:
                z_trans = z_trans.unsqueeze(0)
            rnn_input = torch.cat([input_embedding, z_trans], 2)
            # rnn_input = self.rnn_input_combine(rnn_input)  # Add a combination layer!
            # rnn_input = self.dropout(self.relu(rnn_input))

            if self.rnn_style == 'lstm':
                rnn_output, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))
                hidden = h_t
            elif self.rnn_style == 'gru':
                rnn_output, hidden = self.rnn(rnn_input, hidden)

            rnn_output = self.dropout(rnn_output)  # (input_len, batch, d)

            rnn_outputs_collect.append(rnn_output)
            hidden_collect.append(hidden)

        # if self.context_gate is not None:
        #     outputs = self.context_gate(
        #         emb.view(-1, emb.size(2)),
        #         rnn_output.view(-1, rnn_output.size(2)),
        #         attn_outputs.view(-1, attn_outputs.size(2))
        #     )
        #     outputs = outputs.view(input_len, input_batch, self.hidden_size)
        #     outputs = self.dropout(outputs)
        # else:
        #     outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        rnn_outputs = torch.cat(rnn_outputs_collect, 0)
        hiddens = torch.cat(hidden_collect, 0)
        output_log_softmax = F.log_softmax(self.out(rnn_outputs.view(-1, rnn_outputs.size(2))), dim=1)   # ???
        output_log_softmax = output_log_softmax.view(rnn_outputs.size(0), rnn_outputs.size(1), -1)

        # output_log_sofmax: str_len x batch_size x vocab_size
        # hiddens: str_len x batch_size x hidden_size
        # Padding
        if one_step == False and padding == True:
            output_log_softmax = F.pad(output_log_softmax, (0, 0, 0, 0, 1, 0))
            hiddens = F.pad(hiddens, (0, 0, 0, 0, 1, 0))

        if self._batch_first:
            output_log_softmax = output_log_softmax.transpose(0, 1)
            hiddens = hiddens.transpose(0, 1)
        return output_log_softmax, hiddens

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

class vinilla_seq2seq(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, attn_type = 'dot', n_layers=1, dropout_p=0.1,
                 max_length=MAX_LENGTH, use_cuda=True, batch_first=False, padding_idx=1, context_gate=None,
                 use_bidirectional=False, encoder_n_layer = 1, decoder_n_layer = 1, decoder_rnn_style = 'gru', encoder_rnn_style = 'gru',
                 use_attention = True, use_output_layer = False, use_first_encoder_hidden_state = False, bow_loss = False):
        super(vinilla_seq2seq, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_type = attn_type
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.batch_size = batch_size
        self.direction_num = int(use_bidirectional) + 1
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.decoder_rnn_style = decoder_rnn_style
        self.encoder_rnn_style = encoder_rnn_style
        self.use_attention = use_attention
        self.bow_loss = bow_loss

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding
        self.encoder = EncoderRNN(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  use_cuda=self.use_cuda,
                                  batch_first=batch_first,
                                  embedding_model=self.embedding,
                                  drop_out=dropout_p,
                                  use_bidirectional=use_bidirectional,
                                  n_layers = encoder_n_layer,
                                  rnn_style=encoder_rnn_style)
        if attn_type == 'dot':
            self.attention = DotAttention(hidden_size, use_cuda=use_cuda)
        elif attn_type == 'general':
            self.attention = GeneralAttention(hidden_size, use_cuda=use_cuda)
        if self.use_attention:
            self.decoder = AttnDecoderRNN(embedding_size,
                                          hidden_size,
                                          vocab_size,
                                          encoder_layer=encoder_n_layer,
                                          attn_obj = self.attention,
                                          embedding_model = self.embedding,
                                          use_cuda=use_cuda,
                                          batch_first=batch_first,
                                          dropout_p=dropout_p,
                                          max_length=max_length,
                                          use_bidiretional_encoder=use_bidirectional,
                                          n_layers=decoder_n_layer,
                                          rnn_style=decoder_rnn_style,
                                          use_output_layer=use_output_layer,
                                          use_first_encoder_hidden_state=use_first_encoder_hidden_state,
                                          output_bow_loss=bow_loss)
        else:
            self.decoder = DecoderRNN(embedding_size,
                                          hidden_size,
                                          vocab_size,
                                          encoder_layer=encoder_n_layer,
                                          embedding_model=self.embedding,
                                          use_cuda=use_cuda,
                                          batch_first=batch_first,
                                          dropout_p=dropout_p,
                                          max_length=max_length,
                                          use_bidiretional_encoder=use_bidirectional,
                                          n_layers=decoder_n_layer,
                                          rnn_style=decoder_rnn_style,)
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        print('use cuda: ', self.use_cuda)
        print('bi-directional', use_bidirectional)

    def forward(self, input_idx, tgt_idx, context_lengths):
        encoder_input = input_idx
        decoder_input = tgt_idx
        encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths)
        output_log_softmax, hidden, attns,  = \
            self.decoder(decoder_input, encoder_hs, torch.LongTensor(context_lengths), encoder_ht)
        # if self.bow_loss:
        #     (output_log_softmax, rnn_outputs) = output_log_softmax
        #     if self.training:
        #         return (output_log_softmax, rnn_outputs)
        #     else:
        #         return output_log_softmax
        return output_log_softmax

class my_loss(nn.Module):
    def __init__(self, loss_type='NLLLoss', ignore_index = 0, use_cuda = False, max_len = 128,
                 bow_loss = False, pp_knowledge = None):
        super(my_loss, self).__init__()
        self._ignore_index = ignore_index
        if (loss_type == 'NLLLoss'):
            self.criterion = torch.nn.NLLLoss(weight=None,
                                     size_average=True,
                                     ignore_index=ignore_index)
        self.use_cuda = use_cuda
        self._max_len  = max_len
        self.bow_loss = bow_loss
        self.pp_knowledge = pp_knowledge

    def forward(self, log_softmax_output, target_output, context_lengths = None,
                rnn_output = None, bow_loss_weight = 0.0, is_train = True):
        if context_lengths is not None:
            mask = sequence_mask(torch.LongTensor(context_lengths), max_len=target_output.size()[1])
            if self.use_cuda == True:
                mask = mask.cuda()
            if self.bow_loss and is_train:
                mask_for_rnn_output = mask.unsqueeze(2).expand(log_softmax_output.size())
            # print(mask.size())
            # print(mask_for_rnn_output.size())
            target_output.data.masked_fill_(1 - mask, -int(self._ignore_index)) # Mask with ignore_index
            num_of_elements = mask.sum() - mask.size(0)

        loss = self.criterion(log_softmax_output.contiguous().view(-1, log_softmax_output.size(2)),
                              target_output.view(-1))

        if is_train == False:
            return loss

        if self.bow_loss:
            # mask_t = Variable(torch.FloatTensor([[1,1,0], [1,0,0]]))
            # print(mask_t)
            # print(mask_t.unsqueeze(2).expand(2,3,4))
            # print(rnn_output.size())
            epsilon = 1e-6
            rnn_output.data.masked_fill_(1 - mask_for_rnn_output, 0)
            bow_probs = (F.sigmoid(torch.sum(rnn_output, dim=1)) + epsilon).log()
            # print(bow_probs)
            if self.pp_knowledge != None:
                bow_loss = - torch.sum(bow_probs * sparse_to_matrix(target_output, bow_probs.size(-1),
                                                                    self.use_cuda, use_paraphrase_knowledge = True, word2pwords = self.pp_knowledge)) / num_of_elements  # batch_size
            else:
                bow_loss = - torch.sum(bow_probs * sparse_to_matrix(target_output, bow_probs.size(-1),
                                 self.use_cuda)) / num_of_elements # batch_size
            return loss + bow_loss_weight * bow_loss, bow_loss

        return loss

class vae_loss(nn.Module):
    def __init__(self, batch_size, loss_type='NLLLoss', ignore_index = 0, use_cuda = False, max_len = 128, KLD_weight=1.0,
                  bow_loss = False, bow_embed_loss = False):
        super(vae_loss, self).__init__()
        self._ignore_index = ignore_index
        if (loss_type == 'NLLLoss'):
            self.criterion = torch.nn.NLLLoss(weight=None,
                                     size_average=True,  #True!
                                     ignore_index=ignore_index)
        self.use_cuda = use_cuda
        self._max_len  = max_len
        self.batch_size = batch_size
        self.KLD_weight = KLD_weight
        self.bow_loss = bow_loss

    def KL_annealing(self, klw):
        self.KLD_weight = klw

    def forward(self, log_softmax_output, target_output,
                mu = 0, logvar = 1, mu_prior = 0, logvar_prior = 1, context_lengths = None,
                is_train = True, mu_ns = 0, logvar_ns = 0, is_ns = False, bow_log_softmax = None):
        if is_train == True and isinstance(mu_prior, int) and mu_prior == 0 and isinstance(logvar_prior, int) and\
                logvar_prior == 1:
            mu_prior = Variable(torch.zeros(mu.size()))
            logvar_prior = Variable(torch.zeros(logvar.size()))
            if torch.cuda.is_available():
                mu_prior = mu_prior.cuda()
                logvar_prior = logvar_prior.cuda()

        if context_lengths is not None:
            mask = sequence_mask(torch.LongTensor(context_lengths), max_len=target_output.size()[1])
            if self.use_cuda == True:
                mask = mask.cuda()
            target_output.data.masked_fill_(1 - mask, -int(self._ignore_index)) # Mask with ignore_index

        loss = self.criterion(log_softmax_output.contiguous().view(-1, log_softmax_output.size(-1)),
                              target_output.view(-1))
        num_of_elements = mask.sum() - mask.size(0)

        if is_train == False:
            return loss, 0
        else:
            if logvar.dim() == 2:
                logvar = logvar.unsqueeze(1)
                logvar_prior = logvar_prior.unsqueeze(1)
                mu_prior = mu_prior.unsqueeze(1)
                mu = mu.unsqueeze(1)
            # KL
            # D(N(\mu_0, \sigma_0) || N(\mu_1, \sigma_1)) =
            # 0.5 * (tr(\sigma_1^ -1 * \sigma_0) + (\mu_1 - \mu_0)^T * \sigma_1^-1 * (\mu_1 - \mu_0)
            #  - k + log(det(\sigma_1) / det(\sigma_0)))
            KLD = 0.5 * (
                torch.sum(logvar.exp().div(logvar_prior.exp()))
                + torch.sum(torch.bmm((mu_prior - mu), ((mu_prior - mu) / logvar_prior.exp()).transpose(1,2)))
                - mu.size(2) * mu.size(0) * mu.size(1)
                + torch.sum(logvar_prior - logvar)
            )

            # Normalise by same number of elements as in reconstruction
            KLD /= num_of_elements

            if self.bow_loss:
                bow_loss = torch.sum(bow_log_softmax.squeeze() * sparse_to_matrix(target_output, bow_log_softmax.size(-1),
                                 self.use_cuda)) / num_of_elements
            else:
                bow_loss = 0

            return loss + self.KLD_weight * KLD - 1.0 * bow_loss, KLD, bow_loss

# (Gupta, 2017)
class vinilla_vae(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, z_size, attn_type = 'dot', n_layers=1, dropout_p=0.1,
                 max_length=MAX_LENGTH, use_cuda=True, batch_first=False, padding_idx=1, context_gate=None,
                 use_bidirectional=False, encoder_n_layer = 1, decoder_n_layer = 1, decoder_rnn_style = 'gru', encoder_rnn_style = 'gru',
                 share_encoder = False, decoder_input_dropout = 0.0, vae_first_h = True, bow_loss = True, vae_attention = False,
                 vae_attention_method = 'share', teacher_forcing_rate = 0.0):
        super(vinilla_vae, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_type = attn_type
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.batch_size = batch_size
        self.direction_num = int(use_bidirectional) + 1
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.decoder_rnn_style = decoder_rnn_style
        self.encoder_rnn_style = encoder_rnn_style
        self.share_encoder = share_encoder
        self.decoder_input_dropout = decoder_input_dropout
        self.bow_loss = bow_loss
        self.use_vae_attention = vae_attention
        self.teacher_forcing_rate = teacher_forcing_rate

        self.z_size = z_size   # hidden variable space dimension

        # VAE part
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(self.hidden_size, self.z_size)
        self.fc_var = nn.Linear(self.hidden_size, self.z_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding
        # VAE Attention layer:
        if self.use_vae_attention:
            self.vae_attention = VaeAttention(hidden_size, hidden_size, hidden_size, use_cuda,
                                              attn_method=vae_attention_method)
        else:
            self.vae_attention = None
        if share_encoder:
            self.encoder = EncoderRNN(vocab_size=vocab_size,
                                      embedding_size=embedding_size,
                                      hidden_size=hidden_size,
                                      use_cuda=self.use_cuda,
                                      batch_first=batch_first,
                                      embedding_model=self.embedding,
                                      drop_out=dropout_p,
                                      use_bidirectional=use_bidirectional,
                                      n_layers = encoder_n_layer,
                                      rnn_style=self.encoder_rnn_style)
        else:
            self.encoder1 = EncoderRNN(vocab_size=vocab_size,
                                      embedding_size=embedding_size,
                                      hidden_size=hidden_size,
                                      use_cuda=self.use_cuda,
                                      batch_first=batch_first,
                                      embedding_model=self.embedding,
                                      drop_out=dropout_p,
                                      use_bidirectional=use_bidirectional,
                                      n_layers=encoder_n_layer,
                                        rnn_style=self.encoder_rnn_style)
            self.encoder2 = EncoderRNN(vocab_size=vocab_size,
                                       embedding_size=embedding_size,
                                       hidden_size=hidden_size,
                                       use_cuda=self.use_cuda,
                                       batch_first=batch_first,
                                       embedding_model=self.embedding,
                                       drop_out=dropout_p,
                                       use_bidirectional=use_bidirectional,
                                       n_layers=encoder_n_layer,
                                       rnn_style=self.encoder_rnn_style)
        if attn_type == 'dot':
            self.attention = DotAttention(hidden_size, use_cuda=use_cuda)
        self.decoder1 = AttnDecoderRNN(embedding_size,
                                      hidden_size,
                                      vocab_size,
                                      encoder_layer=encoder_n_layer,
                                      attn_obj=self.attention,
                                      embedding_model=self.embedding,
                                      use_cuda=use_cuda,
                                      batch_first=batch_first,
                                      dropout_p=dropout_p,
                                      max_length=max_length,
                                      use_bidiretional_encoder=use_bidirectional,
                                      n_layers=decoder_n_layer,
                                      rnn_style=decoder_rnn_style)
        self.decoder2 = AttnDecoderRNN_vae(embedding_size,
                                           hidden_size,
                                           vocab_size,
                                           z_size,
                                           encoder_layer=encoder_n_layer,
                                           attn_obj=self.attention,
                                           embedding_model=self.embedding,
                                           use_cuda=use_cuda,
                                           batch_first=batch_first,
                                           dropout_p=dropout_p,
                                           max_length=max_length,
                                           use_bidiretional_encoder=use_bidirectional,
                                           n_layers=decoder_n_layer,
                                           rnn_style=decoder_rnn_style,
                                           decoder_input_dropout=decoder_input_dropout,
                                           vae_first_h=vae_first_h,
                                           use_vae_attention=self.use_vae_attention,
                                           vae_attention=self.vae_attention,
                                           teacher_forcing_rate=teacher_forcing_rate
                                           )
        if self.bow_loss:
            self.mlp_bow = nn.Linear(self.z_size, self.vocab_size)



        if self.use_cuda:
            if self.share_encoder:
                self.encoder = self.encoder.cuda()
            else:
                self.encoder1 = self.encoder1.cuda()
                self.encoder2 = self.encoder2.cuda()
            self.decoder1 = self.decoder1.cuda()
            self.decoder2 = self.decoder2.cuda()
            self.fc_var = self.fc_var.cuda()
            self.fc_mu = self.fc_mu.cuda()
            if self.bow_loss:
                self.mlp_bow = self.mlp_bow.cuda()

        torch.nn.init.xavier_uniform(self.fc_mu.weight)
        torch.nn.init.xavier_uniform(self.fc_var.weight)

        if share_encoder:
            self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        else:
            self.encoder_hidden = self.encoder1.initHidden(self.batch_size)
        print('use cuda: ', self.use_cuda)
        print('bi-directional', use_bidirectional)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def encoding(self, input_idx, tgt_idx, context_lengths_src, context_lengths_tgt, cal_tgt=True):
        encoder_input = input_idx
        decoder_input = tgt_idx

        if self.share_encoder:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)
        else:
            encoder_hs, encoder_ht = self.encoder1(encoder_input, self.encoder_hidden, context_lengths_src)
        output_log_softmax, hidden, attns = \
            self.decoder1(decoder_input, encoder_hs, torch.LongTensor(context_lengths_src), encoder_ht)

        if self.training:
            if self._batch_first:
                h_t = hidden.transpose(0,1)[-1]
            else:
                h_t = hidden[-1]
            z_mu = self.fc_mu(h_t)
            z_logvar = self.fc_var(h_t)
            # z_mu = self.relu(self.fc_mu(h_t))
            # z_logvar = self.relu(self.fc_var(h_t))
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = Variable(torch.zeros(self.batch_size, self.z_size).normal_())
            z = z.cuda() if self.use_cuda else z

        # VAE Decoding part:
        if self.share_encoder:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths)
        else:
            encoder_hs, encoder_ht = self.encoder2(encoder_input, self.encoder_hidden, context_lengths)
        output_log_softmax, hidden, attns = \
            self.decoder2(decoder_input, encoder_hs, torch.LongTensor(context_lengths), encoder_ht, z.unsqueeze(1))

        return {'representation_src': z_mu_prior, 'representation_tgt': z_mu_post}


    def forward(self, input_idx, tgt_idx, context_lengths):
        encoder_input = input_idx
        decoder_input = tgt_idx

        # VAE encoding part:
        if self.training:
            if self.share_encoder:
                encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths)
            else:
                encoder_hs, encoder_ht = self.encoder1(encoder_input, self.encoder_hidden, context_lengths)
            output_log_softmax, hidden, attns = \
                self.decoder1(decoder_input, encoder_hs, torch.LongTensor(context_lengths), encoder_ht)

        # VAE
        if self.training:
            if self._batch_first:
                h_t = hidden.transpose(0,1)[-1]
            else:
                h_t = hidden[-1]
            z_mu = self.fc_mu(h_t)
            z_logvar = self.fc_var(h_t)
            # z_mu = self.relu(self.fc_mu(h_t))
            # z_logvar = self.relu(self.fc_var(h_t))
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = Variable(torch.zeros(self.batch_size, self.z_size).normal_())
            z = z.cuda() if self.use_cuda else z

        # VAE Decoding part:
        if self.share_encoder:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths)
        else:
            encoder_hs, encoder_ht = self.encoder2(encoder_input, self.encoder_hidden, context_lengths)
        output_log_softmax, hidden, attns = \
            self.decoder2(decoder_input, encoder_hs, torch.LongTensor(context_lengths), encoder_ht, z.unsqueeze(1))

        if self.bow_loss:
            bow_log_softmax = F.log_softmax(self.mlp_bow(z))

        if self.training:
            if self.bow_loss:
                return output_log_softmax, z_mu, z_logvar, bow_log_softmax
            else:
                return output_log_softmax, z_mu, z_logvar, None
        else:
            return output_log_softmax

# (Zhang, 2016)
class vnmt(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, z_size, attn_type = 'dot', n_layers=1, dropout_p=0.1,
                 max_length=MAX_LENGTH, use_cuda=True, batch_first=False, padding_idx=1, context_gate=None,
                 use_bidirectional=False, encoder_n_layer = 1, decoder_n_layer = 1, decoder_rnn_style = 'gru', encoder_rnn_style = 'gru',
                 mean_pooling = False, pre_training = False, use_attention = True, active_function = 'None', vae_attention = False,
                 vae_attention_method = 'share', z_sample_num = 1, cat_for_post = True, bow_loss = False, vae_first_embedding = False,
                 use_decoder_encoding = True, share_encoder = False, vae_first_h = True, batch_normalize = True
                 ):
        super(vnmt, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_type = attn_type
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.batch_size = batch_size
        self.direction_num = int(use_bidirectional) + 1
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.decoder_rnn_style = decoder_rnn_style
        self.encoder_rnn_style = encoder_rnn_style
        self.mean_pooling = mean_pooling
        self.pre_training = pre_training
        self.use_attention = use_attention
        self.active_function = active_function
        self.use_vae_attention = vae_attention
        self.z_sample_number = z_sample_num
        self.cat_for_post = cat_for_post   # If don't cat encoder and docoder representation.
        self.bow_loss = bow_loss # If use bag of words loss
        self.vae_first_embedding = vae_first_embedding  # use the z (maybe through a transformation) as the first "word" embedding in decoder.
        self.use_decoder_encoding = use_decoder_encoding  # use decoder in vae encoding part.
        self.use_vinilla_decoder = vae_first_embedding  # If we use the vae first embedding method, we can just use the vinilla decoder.
        self.share_encoder = share_encoder
        self.vae_first_h = vae_first_h
        self.batch_normalize = batch_normalize
        if batch_normalize:
            self.bn = nn.BatchNorm1d(z_size, affine=True) # without affine

        self.z_size = z_size   # hidden variable space dimension

        # VAE part
        if active_function == 'relu':
            self.active_func = nn.ReLU()
        elif active_function == 'tanh':
            self.active_func = torch.tanh()
        elif active_function == 'sigmoid':
            self.active_func = nn.Sigmoid()
        elif active_function == 'None':
            pass
        else:
            raise ValueError("no active function name: '" + active_function + "'")

        if self.use_decoder_encoding == True:
            post_dim = self.hidden_size * self.direction_num + self.hidden_size
        else:
            post_dim = self.hidden_size * self.direction_num * 2
        if not self.cat_for_post:
            post_dim -= self.hidden_size * self.direction_num

        self.prior_mu = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        self.prior_var = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        self.post_mu = nn.Linear(post_dim, self.z_size)
        self.post_var = nn.Linear(post_dim, self.z_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding
        self.encoder = EncoderRNN(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  use_cuda=self.use_cuda,
                                  batch_first=batch_first,
                                  embedding_model=self.embedding,
                                  drop_out=dropout_p,
                                  use_bidirectional=use_bidirectional,
                                  n_layers = encoder_n_layer,
                                  rnn_style=self.encoder_rnn_style)
        if not share_encoder:
            self.encoder_encoding = EncoderRNN(vocab_size=vocab_size,
                                               embedding_size=embedding_size,
                                               hidden_size=hidden_size,
                                               use_cuda=self.use_cuda,
                                               batch_first=batch_first,
                                               embedding_model=self.embedding,
                                               drop_out=dropout_p,
                                               use_bidirectional=use_bidirectional,
                                               n_layers = encoder_n_layer,
                                               rnn_style=self.encoder_rnn_style)

        if attn_type == 'dot':
            self.attention = DotAttention(hidden_size, use_cuda=use_cuda)
        elif attn_type == 'general':
            self.attention = GeneralAttention(hidden_size, use_cuda=use_cuda)

        # VAE Attention layer:
        if self.use_vae_attention:
            self.vae_attention = VaeAttention(hidden_size, hidden_size, hidden_size, use_cuda, attn_method=vae_attention_method)
        else:
            self.vae_attention = None

        if self.use_vinilla_decoder:
            self.decoder = AttnDecoderRNN(embedding_size,
                                          hidden_size,
                                          vocab_size,
                                          encoder_layer=encoder_n_layer,
                                          attn_obj=self.attention,
                                          embedding_model=self.embedding,
                                          use_cuda=use_cuda,
                                          batch_first=batch_first,
                                          dropout_p=dropout_p,
                                          max_length=max_length,
                                          use_bidiretional_encoder=use_bidirectional,
                                          n_layers=decoder_n_layer,
                                          rnn_style=decoder_rnn_style,
                                          use_output_layer=False,
                                          use_first_encoder_hidden_state=False
                                          )
        else:
            if self.use_attention:
                self.decoder = AttnDecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  attn_obj=self.attention,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style,
                                                  use_vae_attention=self.use_vae_attention,
                                                  vae_attention=self.vae_attention,
                                                  vae_first_h=vae_first_h)
            else:
                self.decoder = DecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style)
        if use_decoder_encoding:
            self.decoder_encoding = AttnDecoderRNN(embedding_size,
                                              hidden_size,
                                              vocab_size,
                                              encoder_layer=encoder_n_layer,
                                              attn_obj=self.attention,
                                              embedding_model=self.embedding,
                                              use_cuda=use_cuda,
                                              batch_first=batch_first,
                                              dropout_p=dropout_p,
                                              max_length=max_length,
                                              use_bidiretional_encoder=use_bidirectional,
                                              n_layers=decoder_n_layer,
                                              rnn_style=decoder_rnn_style)
        else:
            self.decoder_encoding = None

        if self.bow_loss:
            self.mlp_bow = nn.Linear(self.z_size, self.vocab_size)

        if self.vae_first_embedding:
            self.z2embedding = nn.Linear(self.z_size, self.embedding_size)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            if use_decoder_encoding:
                self.decoder_encoding = self.decoder_encoding.cuda()
            if not share_encoder:
                self.encoder_encoding = self.encoder_encoding.cuda()
            self.prior_mu = self.prior_mu.cuda()
            self.prior_var = self.prior_var.cuda()
            self.post_mu = self.post_mu.cuda()
            self.post_var = self.post_var.cuda()
            if self.bow_loss:
                self.mlp_bow = self.mlp_bow.cuda()
            if self.vae_first_embedding:
                self.z2embedding = self.z2embedding.cuda()
            if self.batch_normalize:
                self.bn = self.bn.cuda()

        # init
        torch.nn.init.xavier_uniform(self.prior_mu.weight)
        torch.nn.init.xavier_uniform(self.prior_var.weight)
        torch.nn.init.xavier_uniform(self.post_mu.weight)
        torch.nn.init.xavier_uniform(self.post_var.weight)
        if self.batch_normalize:
            torch.nn.init.constant(self.bn.weight, 1.0)
            torch.nn.init.constant(self.bn.bias, 0.0)

        self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        print('use cuda: ', self.use_cuda)
        print('bi-directional', use_bidirectional)

    def stop_vae_pre_train(self):
        self.decoder.vae_pre_train = False

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

        # if self.training:
        #   std = logvar.mul(0.5).exp_()
        #   eps = Variable(std.data.new(std.size()).normal_())
        #   return eps.mul(std).add_(mu)
        # else:
        #   return mu

    def pre_train(self):
        self.pre_training = True
    def stop_pre_train(self):
        self.pre_training = False

    def encoding(self, input_idx, tgt_idx, context_lengths_src, context_lengths_tgt, cal_tgt=True):
        encoder_input = input_idx
        decoder_input = tgt_idx

        if self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)
        else:
            encoder_hidden = self.encoder_encoding.initHidden(self.batch_size)
            encoder_hs_src, encoder_ht_src_ori = self.encoder_encoding(encoder_input, encoder_hidden, context_lengths_src)
        encoder_ht_src = torch.chunk(encoder_ht_src_ori, self.encoder_n_layer, dim=1)[-1]  # Only consider the last layer!
        encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)

        if self.mean_pooling:
            mean_pooling_prior = torch.sum(encoder_hs_src, 1)
            mean_pooling_prior = (
                    mean_pooling_prior / Variable(torch.FloatTensor(context_lengths_src)).unsqueeze(1).expand_as(
                mean_pooling_prior).cuda()).unsqueeze(1)
            z_mu_prior = self.prior_mu(mean_pooling_prior)
        else:
            z_mu_prior = self.prior_mu(encoder_ht_src)

        encoder_hs_tgt, encoder_ht_tgt_ori = self.encoder(decoder_input, self.encoder_hidden,
                                                          context_lengths_tgt, sort=False)
        encoder_ht_tgt = torch.chunk(encoder_ht_tgt_ori, self.encoder_n_layer, dim=1)[
            -1]  # Only consider the last layer!
        encoder_ht_tgt = encoder_ht_tgt.contiguous().view(encoder_ht_tgt.size(0), 1, -1)

        if self.mean_pooling:
            mean_pooling_post = torch.sum(encoder_hs_tgt, 1)
            mean_pooling_post = (mean_pooling_post / Variable(torch.FloatTensor(context_lengths_tgt)).unsqueeze(1).expand_as(mean_pooling_post).cuda()).unsqueeze(1)
            z_mu_post = self.prior_mu(mean_pooling_post)
        else:
            z_mu_post = self.prior_mu(encoder_ht_tgt)

        return {'representation_src': z_mu_prior, 'representation_tgt': z_mu_post}


    def forward(self, input_idx, tgt_idx, context_lengths_src, context_lengths_tgt):
        encoder_input = input_idx
        decoder_input = tgt_idx

        # Pre-training:
        if self.pre_training:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src, sort = True)
            output_log_softmax, hidden, attns = \
                self.decoder_encoding(decoder_input, encoder_hs, torch.LongTensor(context_lengths_src),
                                      encoder_ht)
            return output_log_softmax

        # VAE encoding part:
        if self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)
        else:
            encoder_hidden = self.encoder_encoding.initHidden(self.batch_size)
            encoder_hs_src, encoder_ht_src_ori = self.encoder_encoding(encoder_input, encoder_hidden, context_lengths_src)
        encoder_ht_src = torch.chunk(encoder_ht_src_ori, self.encoder_n_layer, dim=1)[-1]  # Only consider the last layer!
        encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)

        if self.training:
            if self.use_decoder_encoding:
                output_log_softmax, hidden, attns = \
                    self.decoder_encoding(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori)
                if self._batch_first:
                    encoder_ht_tgt = hidden.transpose(0, 1)[-1]
                else:
                    encoder_ht_tgt = hidden[-1]
                encoder_ht_tgt = encoder_ht_tgt.unsqueeze(1)
            else:
                encoder_hs_tgt, encoder_ht_tgt_ori = self.encoder(decoder_input, self.encoder_hidden,
                                                                  context_lengths_tgt, sort = False)
                encoder_ht_tgt = torch.chunk(encoder_ht_tgt_ori, self.encoder_n_layer, dim=1)[
                    -1]  # Only consider the last layer!
                encoder_ht_tgt = encoder_ht_tgt.contiguous().view(encoder_ht_tgt.size(0), 1, -1)


        # VAE
        if self.mean_pooling:
            mean_pooling_prior = torch.sum(encoder_hs_src, 1)
            mean_pooling_prior = (mean_pooling_prior / Variable(torch.FloatTensor(context_lengths_src)).unsqueeze(1).expand_as(mean_pooling_prior).cuda()).unsqueeze(1)
            z_mu_prior = self.prior_mu(mean_pooling_prior)
            z_logvar_prior = self.prior_var(mean_pooling_prior)
        else:
            z_mu_prior = self.prior_mu(encoder_ht_src)
            z_logvar_prior = self.prior_var(encoder_ht_src)
        if self.active_function != 'None':
            z_mu_prior = self.active_func(z_mu_prior)
            z_logvar_prior = self.active_func(z_logvar_prior)

        z_s = []
        if self.training:
            if self.cat_for_post:
                post_input = torch.cat([encoder_ht_tgt, encoder_ht_src], dim=2)
            else:
                post_input = encoder_ht_tgt
            if self.mean_pooling:
                mean_pooling_post = torch.sum(encoder_hs_tgt, 1)
                mean_pooling_post = (mean_pooling_post / Variable(torch.FloatTensor(context_lengths_tgt)).unsqueeze(
                    1).expand_as(mean_pooling_post).cuda()).unsqueeze(1)
                post_input = torch.cat([mean_pooling_post, mean_pooling_prior], dim=2) if self.cat_for_post else mean_pooling_post

            z_mu_post = self.post_mu(post_input)
            z_logvar_post = self.post_var(post_input)
            if self.active_function != 'None':
                z_mu_post = self.active_func(z_mu_post)
                z_logvar_post = self.active_func(z_logvar_post)

            for i in range(self.z_sample_number):
                z = self.reparameterize(z_mu_post, z_logvar_post)
                z_s.append(z)
        else:
            for i in range(self.z_sample_number):
                z = self.reparameterize(z_mu_prior, z_logvar_prior)
                z_s.append(z)

        # VAE Decoding part:
        if not self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)

        output_log_softmax_s = []
        for i in range(self.z_sample_number):
            z = z_s[i]
            if self.vae_first_embedding:
                z_embedding = self.z2embedding(z)

            if self.use_vinilla_decoder:
                output_log_softmax, hidden, _ = \
                    self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src),
                                 encoder_ht_src_ori, use_z_embedding=True, z_embedding=z_embedding)
            else:
                if self.use_attention:
                    output_log_softmax, hidden, _ = \
                        self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z)
                else:
                    output_log_softmax, hidden = \
                        self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z)
            output_log_softmax_s.append(output_log_softmax)

        if (self.z_sample_number > 1):
            output_log_softmax = torch.exp(output_log_softmax_s[0])
            for i in range(0, self.z_sample_number):
                output_log_softmax = output_log_softmax + torch.exp(output_log_softmax_s[i])
            output_log_softmax/=self.z_sample_number
            output_log_softmax.log_()
        else:
            output_log_softmax = output_log_softmax_s[0]
            if self.bow_loss:
                z = z_s[0]
                bow_log_softmax =  F.log_softmax(self.mlp_bow(z))


        if self.training:
            if self.bow_loss:
                return output_log_softmax, z_mu_prior, z_logvar_prior, z_mu_post, z_logvar_post, bow_log_softmax
            else:
                return output_log_softmax, z_mu_prior, z_logvar_prior, z_mu_post, z_logvar_post
        else:
            return output_log_softmax

# GMM + VAE
class gmmvae(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, z_size, k, attn_type = 'dot', n_layers=1, dropout_p=0.1,
                 max_length=MAX_LENGTH, use_cuda=True, batch_first=False, padding_idx=1, context_gate=None,
                 use_bidirectional=False, encoder_n_layer = 1, decoder_n_layer = 1, decoder_rnn_style = 'gru', encoder_rnn_style = 'gru',
                 mean_pooling = False, pre_training = False, use_attention = True, active_function = 'None', vae_attention = False,
                 vae_attention_method = 'share', z_sample_num = 1, cat_for_post = True, bow_loss = False, vae_first_embedding = False,
                 share_encoder = False, batch_normalize = False, resample_gaussian = False, vae_first_h = True,
                 hidden_reconstruction = False, decoder_use_c = False, unconditional = False, get_c_from_z = True,
                 teacher_forcing_rate=0.0, bos_idx=None, eos_idx = None, pad_idx = None):
        super(gmmvae, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_type = attn_type
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.batch_size = batch_size
        self.direction_num = int(use_bidirectional) + 1
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.decoder_rnn_style = decoder_rnn_style
        self.encoder_rnn_style = encoder_rnn_style
        self.mean_pooling = mean_pooling
        self.pre_training = pre_training
        self.use_attention = use_attention
        self.active_function = active_function
        self.use_vae_attention = vae_attention
        self.z_sample_number = z_sample_num
        self.cat_for_post = cat_for_post   # If don't cat encoder and docoder representation.
        self.bow_loss = bow_loss # If use bag of words loss
        self.vae_first_embedding = vae_first_embedding  # use the z (maybe through a transformation) as the first "word" embedding in decoder.
        self.use_vinilla_decoder = vae_first_embedding  # If we use the vae first embedding method, we can just use the vinilla decoder.
        self.share_encoder = share_encoder
        self.k = k
        self.rsp_gaussian = resample_gaussian
        self.mean_pooling = False
        self.batch_normalize = batch_normalize
        self.vae_first_h = vae_first_h
        self.hidden_reconstruction = hidden_reconstruction
        self.decoder_use_c = decoder_use_c # todo: how to sampling?
        self.unconditional = unconditional
        self.teacher_forcing_rate = teacher_forcing_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        if get_c_from_z:
            self.get_c_from_z = lambda z, r : z
        else:
            self.get_c_from_z = lambda z, r : r

        if self.decoder_use_c and not self.vae_first_h:
            raise ValueError("Now, you can only set decoder_use_c True when vae_first_h True.")

        self.training_state = 0 # =0: train all, =1: train gaussian from kl terms, =2: train e&d with reconstruction loss and closs

        self.z_size = z_size   # hidden variable space dimension

        # VAE part
        if active_function == 'relu':
            self.active_func = nn.ReLU()
        elif active_function == 'tanh':
            self.active_func = nn.Tanh()
        elif active_function == 'sigmoid':
            self.active_func = nn.Sigmoid()

        if batch_normalize:
            self.bn = nn.BatchNorm1d(z_size, affine=True) # without affine

        # transform hidden state of rnn to hidden variable vector
        self.h2z = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        # add a layer:
        # self.z2z = nn.Linear(self.z_size, self.z_size)
        # self.mu = nn.Linear(self.z_size, self.z_size)
        self.var = nn.Linear(self.z_size, self.z_size)
        # self.var = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        if hidden_reconstruction:
            self.z2h = nn.Linear(self.z_size, self.hidden_size * self.direction_num)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding
        self.encoder = EncoderRNN(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  use_cuda=self.use_cuda,
                                  batch_first=batch_first,
                                  embedding_model=self.embedding,
                                  drop_out=dropout_p,
                                  use_bidirectional=use_bidirectional,
                                  n_layers = encoder_n_layer,
                                  rnn_style=self.encoder_rnn_style)
        if not share_encoder:
            self.encoder_encoding = EncoderRNN(vocab_size=vocab_size,
                                               embedding_size=embedding_size,
                                               hidden_size=hidden_size,
                                               use_cuda=self.use_cuda,
                                               batch_first=batch_first,
                                               embedding_model=self.embedding,
                                               drop_out=dropout_p,
                                               use_bidirectional=use_bidirectional,
                                               n_layers = encoder_n_layer,
                                               rnn_style=self.encoder_rnn_style)

        if unconditional:
            self.attention = None
        else:
            if attn_type == 'dot':
                self.attention = DotAttention(hidden_size, use_cuda=use_cuda)
            elif attn_type == 'general':
                self.attention = GeneralAttention(hidden_size, use_cuda=use_cuda)

        # VAE Attention layer:
        if self.use_vae_attention:
            self.vae_attention = VaeAttention(hidden_size, hidden_size, hidden_size, use_cuda, attn_method=vae_attention_method)
        else:
            self.vae_attention = None

        if self.use_vinilla_decoder:
            self.decoder = AttnDecoderRNN(embedding_size,
                                          hidden_size,
                                          vocab_size,
                                          encoder_layer=encoder_n_layer,
                                          attn_obj=self.attention,
                                          embedding_model=self.embedding,
                                          use_cuda=use_cuda,
                                          batch_first=batch_first,
                                          dropout_p=dropout_p,
                                          max_length=max_length,
                                          use_bidiretional_encoder=use_bidirectional,
                                          n_layers=decoder_n_layer,
                                          rnn_style=decoder_rnn_style,
                                          use_output_layer=False,
                                          use_first_encoder_hidden_state=False
                                          )
        else:
            if self.use_attention:
                self.decoder = AttnDecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  attn_obj=self.attention,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style,
                                                  use_vae_attention=self.use_vae_attention,
                                                  vae_attention=self.vae_attention,
                                                  vae_first_h=vae_first_h,
                                                  decoder_use_c=decoder_use_c,
                                                  use_attention=(not unconditional),
                                                  teacher_forcing_rate=teacher_forcing_rate)
            else:
                self.decoder = DecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style)


        if self.bow_loss:
            self.mlp_bow = nn.Linear(self.z_size, self.vocab_size)

        if self.vae_first_embedding:
            self.z2embedding = nn.Linear(self.z_size, self.embedding_size)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            if not share_encoder:
                self.encoder_encoding = self.encoder_encoding.cuda()
            self.h2z = self.h2z.cuda()
            if self.hidden_reconstruction:
                self.z2h = self.z2h.cuda()
            # self.z2z = self.z2z.cuda()
            self.var = self.var.cuda()
            if self.bow_loss:
                self.mlp_bow = self.mlp_bow.cuda()
            if self.vae_first_embedding:
                self.z2embedding = self.z2embedding.cuda()
            if self.batch_normalize:
                self.bn = self.bn.cuda()

        self.initialization()
        # self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        # self.init_gaussion()
        # torch.nn.init.xavier_uniform(self.h2z.weight)
        # # torch.nn.init.xavier_uniform(self.z2z.weight)
        # torch.nn.init.xavier_uniform(self.var.weight)  # change
        # if self.batch_normalize:
        #     torch.nn.init.xavier_uniform(self.bn.weight)
        #     torch.nn.init.xavier_uniform(self.bn.bias)
        # print('use cuda: ', self.use_cuda)
        # print('bi-directional', use_bidirectional)


    def initialization(self, init_method = "xavier"):
        self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        self.init_gaussion()
        torch.nn.init.xavier_uniform(self.h2z.weight)
        # torch.nn.init.xavier_uniform(self.z2z.weight)
        torch.nn.init.xavier_uniform(self.var.weight)  # change
        if self.hidden_reconstruction:
            torch.nn.init.xavier_uniform(self.z2h.weight)
        if self.batch_normalize:
            torch.nn.init.constant(self.bn.weight, 1.0)
            torch.nn.init.constant(self.bn.bias, 0.0)
            # for key, value in self.bn.named_parameters():
            #     print(key)

            # torch.nn.init.xavier_uniform(self.bn.weight)

    def stop_vae_pre_train(self):
        self.decoder.vae_pre_train = False

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

        # if self.training:
        #   std = logvar.mul(0.5).exp_()
        #   eps = Variable(std.data.new(std.size()).normal_())
        #   return eps.mul(std).add_(mu)
        # else:
        #   return mu

    def pre_train(self):
        self.pre_training = True
    def stop_pre_train(self):
        self.pre_training = False

    def get_ave_h(self, hs, length):
        mask = sequence_mask(torch.LongTensor(length), max_len=hs.size()[1])
        hs_mp = torch.sum(hs * Variable(mask.float()).cuda().unsqueeze(2), dim=1)
        hs_mp = hs_mp / Variable(torch.FloatTensor(length)).cuda().unsqueeze(1)
        return hs_mp

    def cluster_prob(self, query):
        # query: batch x 1 x z_size
        # return: batch x k.
        q_repeat = query.expand(-1, self.k, -1)
        if self.training_state == 2:
            log_p = - ((q_repeat - self.gaussion_mus.detach()) / self.gaussion_vars.detach().abs()).pow(2) / 2 \
                    - self.gaussion_vars.detach().abs().log() - 0.5 * math.log(2*math.pi)
        else:
            log_p = - ((q_repeat - self.gaussion_mus) / self.gaussion_vars.abs()).pow(2) / 2 - self.gaussion_vars.abs().log() - 0.5 * math.log(2*math.pi)
        # print('log_p', log_p)
        log_p_sum = torch.sum(log_p, dim=2)   # sum?
        # print(log_p_sum)
        return F.softmax(log_p_sum, dim=1)

    def init_gaussion(self, init_method = 'randn'):
        # Initialize K gaussion distribution
        # rand / randn
        if init_method == 'randn':
            # gaussion_mus_torch = torch.zeros(self.k, self.z_size)
            gaussion_mus_torch = torch.randn(self.k, self.z_size) # for high dimension
            # gaussion_mus_torch.uniform_(-1, 1)
            gaussion_vars_torch = torch.ones(self.k, self.z_size) * 0.1 # * 0.5 # for low dimension: suggest * 0.1; for higher dimension of z, suggest lager variance.
        elif init_method == 'onehot':
            pass
        # gaussion_vars_torch = torch.ones(self.k, self.z_size)
        if self.use_cuda:
            gaussion_mus_torch = gaussion_mus_torch.cuda()
            gaussion_vars_torch = gaussion_vars_torch.cuda()
        self.gaussion_mus = torch.nn.Parameter(gaussion_mus_torch, requires_grad=True)  # change: False
        self.gaussion_vars = torch.nn.Parameter(gaussion_vars_torch, requires_grad=False)  # change: False

    def resample_gaussian(self, sample_method = 'randn'):
        if sample_method == 'randn':
            self.gaussion_mus.data = torch.randn(self.k, self.z_size).cuda()
            self.gaussion_vars.data = torch.randn(self.k, self.z_size).cuda()
            return

        if self.k == 1:
            if sample_method == "recal":
                eps = self.gaussion_mus.data.new(self.gaussion_mus.size()).normal_()  # k x z_size
                self.gaussion_mus.data = eps.add_(self.gaussion_mus.data)
                eps = self.gaussion_mus.data.new(self.gaussion_mus.size()).normal_()  # k x z_size
                self.gaussion_vars.data = eps.add_(self.gaussion_vars.data)
                # print(self.gaussion_mus)
                # print(self.gaussion_vars)
            else:
                pass
            return
            pass

        new_mu_mu = torch.mean(self.gaussion_mus, dim=0) # z_size
        new_mu_std = torch.mean((self.gaussion_mus - new_mu_mu).pow(), dim=0).pow(0.5) # z_size
        new_var_mu = torch.mean(self.gaussion_vars, dim=0)  # z_size
        new_var_std = torch.mean((self.gaussion_vars - new_var_mu).pow(), dim=0).pow(0.5)  # z_size

        eps = Variable(self.gaussion_mus.data.new(self.gaussion_mus.size()).normal_()) # k x z_size
        self.gaussion_mus.data = eps.mul(new_mu_std.expand(self.k, -1)).add_(new_mu_mu.expand(self.k, -1))
        self.gaussion_vars.data = eps.mul(new_var_std.expand(self.k, -1)).add_(new_var_mu.expand(self.k, -1))

        pass

    def get_c(self, cludis):
        # cludis: batch_size x k
        # return: batch_size x z_size
        if self.decoder_use_c:
            # todo: how to sampling?
            # 1. average mean
            c = torch.mm(cludis, self.gaussion_mus)
        else:
            c = None
        return c

    def encoding(self, encoder_input, decoder_input, context_lengths_src, context_lengths_tgt, h0=None,
                 detach = True, cal_tgt = False, input_is_embedding = False):
        # VAE encoding part:
        if h0 is None:
            h0 = self.encoder_hidden

        if self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, h0, context_lengths_src, input_is_embedding=input_is_embedding)
        else:
            # encoder_hidden = self.encoder_encoding.initHidden(self.batch_size)   # todo: testing?
            encoder_hs_src, encoder_ht_src_ori = self.encoder_encoding(encoder_input, h0,
                                                                       context_lengths_src, input_is_embedding=input_is_embedding)

        encoder_ht_src = torch.chunk(encoder_ht_src_ori, self.encoder_n_layer, dim=1)[
            -1]  # Only consider the last layer!
        encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)
        if not self.unconditional and (self.training or cal_tgt):
            if self.share_encoder:
                encoder_hs_tgt, encoder_ht_tgt_ori = self.encoder(decoder_input, h0, context_lengths_tgt, sort=False)
            else:
                encoder_hs_tgt, encoder_ht_tgt_ori = self.encoder_encoding(decoder_input, h0, context_lengths_tgt, sort=False)

            # _, encoder_ht_tgt_ori = self.encoder(decoder_input, h0, context_lengths_tgt, sort=False)
            encoder_ht_tgt = torch.chunk(encoder_ht_tgt_ori, self.encoder_n_layer, dim=1)[
                -1]  # Only consider the last layer!
            encoder_ht_tgt = encoder_ht_tgt.contiguous().view(encoder_ht_tgt.size(0), 1, -1)

        # if detach:
        #     encoder_ht_src_detach = encoder_ht_src.detach()
        #     # encoder_ht_src_detach = Variable(encoder_ht_src_detach).cuda() if self.use_cuda else Variable(encoder_ht_src_detach)
        #     representation_src = self.h2z(encoder_ht_src_detach)
        # else:
        #     representation_src = self.h2z(encoder_ht_src)
        do_detach = lambda x: x.detach() if detach else x

        if self.mean_pooling:
            encoder_ave_h = self.get_ave_h(do_detach(encoder_hs_src), context_lengths_src)
            representation_src = self.h2z(encoder_ave_h.unsqueeze(1))
        else:
            representation_src = self.h2z(do_detach(encoder_ht_src))
        representation_src = self.active_func(representation_src) if (
                self.active_function != 'None') else representation_src
        # representation_src = self.z2z(representation_src)
        if not self.unconditional and self.training or cal_tgt:
            if self.mean_pooling:
                encoder_ave_h = self.get_ave_h(do_detach(encoder_hs_tgt), context_lengths_tgt)
                representation_tgt = self.h2z(encoder_ave_h.unsqueeze(1))
            else:
                representation_tgt = self.h2z(do_detach(encoder_ht_tgt))
                representation_tgt_detach = self.h2z((encoder_ht_tgt.detach()))

            representation_tgt = self.active_func(representation_tgt) if (
                    self.active_function != 'None') else representation_tgt
            representation_tgt_detach = self.active_func(representation_tgt_detach) if (
                    self.active_function != 'None') else representation_tgt_detach
            # representation_tgt = self.z2z(representation_tgt)

        if self.training_state == 1:
            representation_src = representation_src.detach()
            if self.training:
                representation_tgt = representation_tgt.detach()

        # print(representation_src)
        # print(representation_tgt)

        # Batch Normalization
        if self.batch_normalize:
            representation_src = self.bn(representation_src.squeeze(1)).unsqueeze(1)
            if not self.unconditional and self.training or cal_tgt:
                representation_tgt = self.bn(representation_tgt.squeeze(1)).unsqueeze(1)


        if not self.unconditional and self.training or cal_tgt: #  encoder_ht_src_ori
            return {'encoder_hs_src':encoder_hs_src, 'encoder_ht_src': encoder_ht_src_ori,
                    'representation_src': representation_src, 'representation_tgt': representation_tgt,
                    'encoder_ht_tgt': encoder_ht_tgt, 'representation_tgt_detach': representation_tgt_detach}
        else:
            return {'encoder_hs_src':encoder_hs_src, 'encoder_ht_src': encoder_ht_src_ori,
                    'representation_src': representation_src}

    def get_gaussian_parameters(self, encoding_outputs, fix_logvar = False):
        if not self.unconditional and self.training:
            representation_tgt = encoding_outputs['representation_tgt']
            mu = representation_tgt
            logvar = self.var(representation_tgt)
        else:
            representation_src = encoding_outputs['representation_src']
            # print(representation_src)
            mu = representation_src
            # print(mu)
            logvar = self.var(representation_src)

        if fix_logvar:
            # logvar = Variable(torch.ones(mu.size())).cuda().log()
            logvar = Variable(torch.ones(mu.size()) * 0.1).cuda().log() # mu.size()) * 0.1

        return mu, logvar

    def get_infoloss(self):
        output_log_softmax_collect = []
        context_lengths_src_collect = []

        pad_idx = torch.zeros(self.vocab_size)
        pad_idx[self.pad_idx] = 1
        pad_idx = Variable(pad_idx).unsqueeze(0).unsqueeze(1).cuda()

        for i in range(self.k):
            std = self.gaussion_vars[i, :]
            mu = self.gaussion_mus[i, :]

            # reparameterization:
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            z = z.cuda() if self.use_cuda else z
            z = z.unsqueeze(0) if z.dim() == 1 else z
            z = z.unsqueeze(0) if z.dim() == 2 else z

            # prepare input:
            c = self.get_c(self.cluster_prob(z))
            encoder_ht = self.decoder.initHidden()
            decoder_input = Variable(torch.LongTensor([[self.bos_idx]]))  # input: <s>
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

            # decoding
            sent_length = self.max_length
            bos_idx = torch.zeros(self.vocab_size)
            bos_idx[self.bos_idx] = 1
            output_log_softmax_collect.append(Variable(bos_idx).unsqueeze(0).unsqueeze(1).cuda())

            for di in range(self.max_length - 1):
                first_step = (di == 0)
                output_log_softmax, hidden, attns = self.decoder(decoder_input, None, None, encoder_ht, z, padding=False, first_step=first_step, c=c)
                # [1, 1, vocab_size]

                topv, topi = torch.max(output_log_softmax, dim=-1)
                if topi[0][0].item() == self.eos_idx:
                    sent_length = di + 1
                    break

                decoder_input = topi   # LongTensor of size [beam_search]  -> batch(beam_size) x str_len(=1)

                # get sentence:
                output_log_softmax_collect.append(output_log_softmax)
            for _ in range(sent_length, self.max_length):
                output_log_softmax_collect.append(pad_idx)

            context_lengths_src_collect.append(sent_length)
        output_log_softmax_collect = torch.cat(output_log_softmax_collect, 1).squeeze()
        context_lengths_src_collect = torch.LongTensor(context_lengths_src_collect)

        embedding_matrix = self.embedding(torch.LongTensor(list(range(self.vocab_size))).cuda()) # vocab_size x embedding_size
        embedding_matrix = torch.mm(torch.exp(output_log_softmax_collect), embedding_matrix)  # batch_size x str_len x embedding_size
        embedding_matrix = embedding_matrix.view(self.k, -1, self.embedding_size)

        encoding_outputs = self.encoding(embedding_matrix, None, context_lengths_src_collect, None, input_is_embedding=True,
                                         h0=self.encoder.initHidden(self.k))
        representation_src = encoding_outputs['representation_src']
        cludis_src = self.cluster_prob(representation_src)  # k x k
        return torch.sum(-torch.log(torch.diag(cludis_src, 0) + 1e-8)) / self.k


    def forward(self, input_idx, tgt_idx, context_lengths_src, context_lengths_tgt, src_cls=None,
                tgt_cls=None):
        encoder_input = input_idx
        decoder_input = tgt_idx

        # Pre-training:
        if self.pre_training:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src, sort = True)
            output_log_softmax, hidden, attns = \
                self.decoder_encoding(decoder_input, encoder_hs, torch.LongTensor(context_lengths_src),
                                      encoder_ht)
            return output_log_softmax

        # 1. encoding part:
        encoding_outputs = self.encoding(encoder_input, decoder_input, context_lengths_src, context_lengths_tgt)
        if not self.unconditional and self.training:
            representation_tgt = encoding_outputs['representation_tgt']
        representation_src = encoding_outputs['representation_src']
        encoder_hs_src = encoding_outputs['encoder_hs_src']
        encoder_ht_src_ori = encoding_outputs['encoder_ht_src']

        # print(representation_tgt.size())
        # print(representation_src.size())
        # mean_src = torch.mean(representation_src, dim=0)
        # mean_ht = torch.mean(encoder_ht_src_ori, dim=0)

        # print("mean of src", mean_src)
        # print(self.gaussion_mus)
        # print(self.gaussion_vars)
        # print(representation_tgt)

        # print('variance of src', torch.mean(torch.mean((representation_src - mean_src) * (representation_src - mean_src), dim=0).pow(0.5)).data[0])
        # if not self.unconditional and self.training:
        #     mean_tgt = torch.mean(representation_tgt, dim=0)
        #     # print("mean of tgt", mean_tgt)
        #     # print('variance of src', torch.mean(
        #     # torch.mean((representation_tgt - mean_tgt) * (representation_tgt - mean_tgt), dim=0).pow(0.5)).data[0])
        #     pass

        # 2. calculate q(z|x) / q(z|x, y) and sampling z
        mu, logvar = self.get_gaussian_parameters(encoding_outputs)
        # print(mu)
        z = self.reparameterize(mu, logvar)

        # 3. calculate q(c|x, z) / q(c|x, y, z)
        if not self.unconditional and self.training:
            if tgt_cls is not None:
                cludis_tgt = self.cluster_prob(representation_tgt)
                cludis_tgt_real = torch.zeros(self.batch_size, self.k).scatter_(1, torch.LongTensor(tgt_cls).unsqueeze(1), 1.0)
                cludis_tgt_real = Variable(cludis_tgt_real).cuda() if self.use_cuda else Variable(cludis_tgt_real)
            else:
                cludis_tgt = self.cluster_prob(self.get_c_from_z(z, representation_tgt))


        cludis_src = self.cluster_prob(self.get_c_from_z(z, representation_src))
        # if src_cls is not None:
        #     cludis_src = self.cluster_prob(representation_src)
        #     pass
        #     # cludis_src = torch.zeros(self.batch_size, self.k).scatter_(1, torch.LongTensor(src_cls).unsqueeze(1), 1.0)
        #     # cludis_src = Variable(cludis_src).cuda() if self.use_cuda else Variable(cludis_src)
        # else:
        #     cludis_src = self.cluster_prob(representation_src)

        # print(cludis_src)
        # print(representation_src)
        # print(cludis_src.data.cpu()[0].tolist())
        # print("===")
        # print(representation_src.data.cpu()[0].tolist())
        # print(representation_tgt.data.cpu()[0].tolist())
        # print(self.gaussion_mus)
        # print(self.gaussion_vars)
        # print(cludis_src)
        # topv, topi = cludis_src.data.topk(1)
        # print(topi.cpu().tolist())

        # Test something:
        # test something:
        # inner_product = torch.mm(self.gaussion_mus, self.gaussion_mus.transpose(0, 1))
        # # print(inner_product)
        # length = torch.diag(inner_product, 0)
        # print(length)


        # get c:
        if not self.unconditional and self.training:
            c = self.get_c(cludis_tgt)
        else:
            c = self.get_c(cludis_src)
        # if self.decoder_use_c:
        #     # todo: how to sampling?
        #     # 1. average mean
        #     if self.training:
        #         c = torch.mm(cludis_tgt, self.gaussion_mus)
        #     else:
        #         c = torch.mm(cludis_src, self.gaussion_mus)
        # else:
        #     c = None

        # VAE Decoding part:
        if not self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)

        if self.vae_first_embedding:
            z_embedding = self.z2embedding(z)

        if self.use_vinilla_decoder:
            output_log_softmax, hidden, _ = \
                self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src),
                             encoder_ht_src_ori, use_z_embedding=True, z_embedding=z_embedding)
        else:
            if self.use_attention:
                output_log_softmax, hidden, _ = \
                    self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z, c=c)
            else:
                output_log_softmax, hidden = \
                    self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z, c=c)

        output = {'output_log_softmax': output_log_softmax,
                  'mu': mu,
                  'logvar': logvar,
                  'cludis_tgt': None,
                  'cludis_tgt_real': None,
                  'cludis_src': cludis_src,
                  'bow_loss': None,
                  'gaussion_mus': self.gaussion_mus,
                  'gaussion_vars': self.gaussion_vars
                  }
        if tgt_cls is not None and self.training:
            output['cludis_tgt_real'] = cludis_tgt_real

        if self.bow_loss:
            bow_log_softmax =  F.log_softmax(self.mlp_bow(z))
            output['bow_loss'] = bow_log_softmax

        if not self.unconditional and self.training:
            output['cludis_tgt'] = cludis_tgt

        if self.hidden_reconstruction and self.training:
            # print(self.z2h(encoding_outputs['representation_tgt_detach']))
            # print(encoding_outputs['encoder_ht_tgt'])
            output['hidden_reconstruction_loss'] = \
                torch.sum(torch.sum((self.z2h(encoding_outputs['representation_tgt_detach']) - encoding_outputs['encoder_ht_tgt'].detach()).pow(2), dim=2).pow(0.5)) / self.batch_size
        else:
            output['hidden_reconstruction_loss'] = None

        # if self.training and self.rsp_gaussian:
        #     self.resample_gaussian()

        return output


class gmmvae_loss(nn.Module):
    def __init__(self, batch_size, loss_type='NLLLoss', ignore_index = 0, use_cuda = False, max_len = 128, KLD_weight=1.0,
                  bow_loss = False, cluster_loss = False, closs_lambda = 0.0, hidden_rcs_w=0.0, bow_loss_w = 0.0,
                 mean_zloss = False, unconditional = False):
        super(gmmvae_loss, self).__init__()
        self._ignore_index = ignore_index
        if (loss_type == 'NLLLoss'):
            self.criterion = torch.nn.NLLLoss(weight=None,
                                     size_average=True,  #True!
                                     ignore_index=ignore_index)
        self.use_cuda = use_cuda
        self._max_len  = max_len
        self.batch_size = batch_size
        self.KLD_weight = KLD_weight
        self.bow_loss = bow_loss
        self.cluster_loss = cluster_loss
        self.closs_lambda = closs_lambda
        self.mask_ckl = True
        self.bow_loss_w = bow_loss_w
        self.hidden_rcs_w = hidden_rcs_w
        self.mean_zloss = mean_zloss
        self.unconditional = unconditional

    def KL_annealing(self, klw):
        self.KLD_weight = klw

    def cal_cluster_loss(self, tgt_probs):
        # tgt_probs: batch_size x k
        return torch.sum(torch.mean(tgt_probs, dim=0).pow(2))

    def cluster_kl(self, tgt_probs, src_probs, max_clu_and_detach = False):
        epsilon = 1e-6
        if tgt_probs is None:
            tgt_probs = src_probs
            # tgt_probs = torch.sum(src_probs, dim=0, keepdim=True) / src_probs.size(0)
            # print(tgt_probs.size())

            # uniform distribution
            src_probs = (torch.ones(tgt_probs.size()) / tgt_probs.size(1)).cuda()
            # src_probs = (torch.rand(tgt_probs.size()) / tgt_probs.size(1)).cuda()

            # random distribution: just for test.
            # src_probs = torch.rand(tgt_probs.size())
            # src_probs = src_probs / torch.norm(src_probs, p=2, dim=1, keepdim=True)
            # src_probs = src_probs.cuda()

            # print(src_probs)


            # print(src_probs)
            # print(tgt_probs)
            # src_probs = src_probs.cuda()
        # else:
        tgt_probs = tgt_probs + epsilon

        # tgt_probs = tgt_probs/torch.sum(tgt_probs, dim=1, keepdim=True)
        src_probs = src_probs + epsilon
        # src_probs = src_probs / torch.sum(src_probs, dim=1, keepdim=True)
        # print(tgt_probs, src_probs)
        if not max_clu_and_detach:
            # batch x k
            res = torch.sum(tgt_probs * torch.log(tgt_probs / src_probs), dim=1)  # .log()
        else:
            # pass
            # topv, topi = torch.max(tgt_probs, dim=1)
            # print(topv)
            # msrc_probs = src_probs[range(src_probs.size(0)), topi.detach()]
            # print(msrc_probs)
            res = torch.max(tgt_probs * (tgt_probs / src_probs).log(), dim=1)[0]
            # res = (topv.detach() / msrc_probs).log()
        # print(res.size())
        # print(res < 1e-1)
        res = res * (res>=1e-3).float()
        # print(res)
        # exit()
        return res  # change

    def z_kl(self, gaussion_mus, gaussion_vars, mu, logvar, tgt_probs, max_clu = False,
             detach_prob = False):
        # E_(q(c│x, y) ) KL(q(z | x, y) | | p(z | c))
        # gaussion_mus & gaussion_vars: k x z_dim
        # mu & logvar: batch_size x 1 x z_dim
        # tgt_probs: batch_size x k

        if self.mean_zloss:
            # KL(q(z|x, y) || p(z~ | x, y))
            # z~ = \sum_c q(c|x,y,z) * mu_c
            mean_mean = torch.mm(tgt_probs, gaussion_mus) # batch_size x z_dim
            mean_sigma = torch.mm(tgt_probs, gaussion_vars) # batch_size x z_dim

            kl = 0.5 * (
                    torch.sum(logvar.squeeze().exp().div(mean_sigma.pow(2)), dim=1)
                    + torch.sum((mean_mean - mu.squeeze()).pow(2) / mean_sigma.pow(2), dim=1)
                    - mu.size(1)
                    + torch.sum(mean_sigma.pow(2).log() - logvar.squeeze(), dim=1)
            )  # batch_size x k
            return kl

        k = gaussion_mus.size(0)
        mu_repeat = mu.expand(-1, k, -1)  # batch_size x k x z_dim
        logvar_repeat = logvar.expand(-1, k, -1)
        gaussion_logvars = gaussion_vars.pow(2).log()   # 注意标准差和方差的区别
        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussion_logvars.exp()), dim=2)
                + torch.sum((gaussion_mus - mu_repeat).pow(2) / gaussion_logvars.exp(), dim=2)
                # + torch.sum(torch.bmm((gaussion_mus - mu_repeat), ((gaussion_mus - mu_repeat) / gaussion_logvars.exp()).transpose(1, 2)), dim=2)
                - mu.size(2)
                + torch.sum(gaussion_logvars - logvar_repeat, dim=2)
        )  # batch_size x k
        # print(kl)
        # print(torch.sum(logvar_repeat.exp().div(gaussion_logvars.exp()), dim=2))
        # print(torch.sum((gaussion_mus - mu_repeat).pow(2) / gaussion_logvars.exp(), dim=2))
        # print(tgt_probs)
        if detach_prob:
            # tgt_probs = tgt_probs.detach()
            return torch.sum(kl * tgt_probs.detach(), dim=1)
        if not max_clu: # average kl
            return torch.sum(kl * tgt_probs, dim=1)
        else:
            return torch.max(kl * tgt_probs, dim=1)[0]
            topv, topi = torch.max(tgt_probs, dim=1) # [batch]
            mkl = kl[range(kl.size(0)), topi.detach()] # [batch]
            return mkl  # * topv?
            # return torch.max(kl * tgt_probs, dim=1)[0]

    def forward(self, log_softmax_output, target_output, tgt_probs, src_probs,
                gaussion_mus, gaussion_vars, mu = 0, logvar = 1, context_lengths = None,
                is_train = True, bow_log_softmax = None, standard_loss = False, hidden_reconstruction_loss = None,
                real_tgt_cls = None
                ):
        # if is_train == True and isinstance(mu_prior, int) and mu_prior == 0 and isinstance(logvar_prior, int) and\
        #         logvar_prior == 1:
        #     mu_prior = Variable(torch.zeros(mu.size()))
        #     logvar_prior = Variable(torch.zeros(logvar.size()))
        #     if torch.cuda.is_available():
        #         mu_prior = mu_prior.cuda()
        #         logvar_prior = logvar_prior.cuda()
        # print(mu)
        # print(logvar)


        if context_lengths is not None:
            mask = sequence_mask(torch.FloatTensor(context_lengths), max_len=target_output.size()[1])
            if self.use_cuda == True:
                mask = mask.cuda()
            target_output.data.masked_fill_(1 - mask, -int(self._ignore_index)) # Mask with ignore_index

        loss = self.criterion(log_softmax_output.contiguous().view(-1, log_softmax_output.size(-1)),
                              target_output.view(-1))
        num_of_elements = mask.sum() - mask.size(0)

        if self.bow_loss:
            bow_loss = torch.sum(bow_log_softmax.squeeze() * sparse_to_matrix(target_output, bow_log_softmax.size(-1),
                                                                              self.use_cuda).float()) / (num_of_elements).float()
        else:
            bow_loss = 0

        if is_train == False:
            return loss, 0, 0, 0
        # print(loss, bow_loss)
        # print(gaussion_mus, gaussion_vars)

        if hidden_reconstruction_loss is not None:
            hrloss = hidden_reconstruction_loss
        else:
            hrloss = 0.0

        if standard_loss:
            KLD = 0.5 * (
                    torch.sum(logvar.exp())
                    + torch.sum(torch.bmm(( - mu), (( - mu)).transpose(1, 2)))
                    - mu.size(2) * mu.size(0) * mu.size(1)
                    + torch.sum( - logvar)
            )
            KLD /= num_of_elements
            return loss + self.KLD_weight * KLD, KLD, 0, 0, 0

        ckl = torch.sum(self.cluster_kl(tgt_probs, src_probs)) / src_probs.size(0) #  记得改回来啊
        # if (float(ckl.data[0]) < 0.1):
        #     self.KLD_weight = 0
        if self.cluster_loss:
            if self.unconditional:
                closs = self.cal_cluster_loss(src_probs)
            else:
                closs = self.cal_cluster_loss(tgt_probs)
            # print(closs)
        else:
            closs = 0.0

        # change!!!
        # gaussion_vars = Variable(torch.ones(gaussion_vars.size())).cuda()
        # print(gaussion_vars)
        # CHANGE!!!!!!!
        k = gaussion_mus.size(1)
        # zkl = torch.sum(self.z_kl(gaussion_mus, gaussion_vars, mu, Variable(torch.zeros(logvar.size())).cuda(), tgt_probs)) / self.batch_size
        if real_tgt_cls is not None:
            zkl = torch.sum(self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, real_tgt_cls)) / self.batch_size
        else:
            if self.unconditional:
                zkl = torch.sum(self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, src_probs)) / self.batch_size
            else:
                zkl = torch.sum(self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs)) / self.batch_size

        # print(self.cluster_kl(tgt_probs, src_probs))
        # print(self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs))
        # print('rnn',loss, 'ckl',ckl, 'zkl',zkl)

        # zkl /= (gaussion_mus.size(1))
        # ckl /= (gaussion_mus.size(1))

        # if float(zkl.data[0]) < 0.01:
        #     self.KLD_weight = 0.0

        D = 3.0 # math.fabs(ckl + zkl - D)
        # print(ckl + zkl)
        # print(math.fabs(ckl + zkl - D))

        # tot_loss = loss + self.KLD_weight * (torch.abs(ckl + zkl - D)) + self.closs_lambda * closs - bow_loss * self.bow_loss_w + self.hidden_rcs_w * hrloss
        # tot_loss = loss
        # print(self.closs_lambda)
        # tot_loss = loss + self.closs_lambda * closs - bow_loss * self.bow_loss_w + self.hidden_rcs_w * hrloss
        tot_loss = loss + self.KLD_weight * (ckl + zkl) + self.closs_lambda * closs - bow_loss * self.bow_loss_w + self.hidden_rcs_w * hrloss
        loss_all = {'tot_loss': tot_loss,
                  'rnn_loss': loss,
                  'ckl_loss': ckl,
                  'zkl_loss': zkl,
                  'bow_loss': -bow_loss,
                  "hr_loss": hrloss,
                  'c_loss': closs}
        # print("??")
        return tot_loss, loss_all # bow_loss
        # return loss + self.KLD_weight * (ckl + zkl + closs), ckl, zkl, bow_loss, closs
        # return loss, ckl, zkl, bow_loss
        # return loss + self.KLD_weight * ckl + zkl + bow_loss, ckl, zkl, bow_loss
            # return loss + self.cluster_kl(tgt_probs, src_probs) + self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs) + bow_loss, \
            #        self.cluster_kl(tgt_probs, src_probs), self.z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs), bow_loss

# RVAE
class rvae(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, z_size, attn_type = 'dot', n_layers=1, dropout_p=0.1,
                 max_length=MAX_LENGTH, use_cuda=True, batch_first=False, padding_idx=1, context_gate=None,
                 use_bidirectional=False, encoder_n_layer = 1, decoder_n_layer = 1, decoder_rnn_style = 'gru', encoder_rnn_style = 'gru',
                 mean_pooling = False, pre_training = False, use_attention = True, active_function = 'None', vae_attention = False,
                 vae_attention_method = 'share', z_sample_num = 1, cat_for_post = True, bow_loss = False, vae_first_embedding = False,
                 use_decoder_encoding = True, share_encoder = False,
                 ):
        super(rvae, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._batch_first = batch_first
        self.context_gate = context_gate
        self.batch_size = batch_size
        self.direction_num = int(use_bidirectional) + 1
        self.encoder_n_layer = encoder_n_layer
        self.decoder_n_layer = decoder_n_layer
        self.decoder_rnn_style = decoder_rnn_style
        self.encoder_rnn_style = encoder_rnn_style
        self.mean_pooling = mean_pooling
        self.active_function = active_function
        self.z_sample_number = z_sample_num
        self.cat_for_post = cat_for_post   # If don't cat encoder and docoder representation.
        self.bow_loss = bow_loss # If use bag of words loss
        self.vae_first_embedding = vae_first_embedding  # use the z (maybe through a transformation) as the first "word" embedding in decoder.
        self.use_decoder_encoding = use_decoder_encoding  # use decoder in vae encoding part.
        self.use_vinilla_decoder = vae_first_embedding  # If we use the vae first embedding method, we can just use the vinilla decoder.
        self.share_encoder = share_encoder

        self.z_size = z_size   # hidden variable space dimension

        # VAE part
        if active_function == 'relu':
            self.active_func = nn.ReLU()
        elif active_function == 'tanh':
            self.active_func = nn.Tanh()
        elif active_function == 'sigmoid':
            self.active_func = nn.Sigmoid()
        else:
            raise ValueError("no active function name: '" + active_function + "'")

        if self.use_decoder_encoding == True:
            post_dim = self.hidden_size * self.direction_num + self.hidden_size
        else:
            post_dim = self.hidden_size * self.direction_num * 2
        if not self.cat_for_post:
            post_dim -= self.hidden_size * self.direction_num

        self.prior_mu = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        self.prior_var = nn.Linear(self.hidden_size * self.direction_num, self.z_size)
        self.post_mu = nn.Linear(post_dim, self.z_size)
        self.post_var = nn.Linear(post_dim, self.z_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Share embedding
        self.encoder = EncoderRNN(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  use_cuda=self.use_cuda,
                                  batch_first=batch_first,
                                  embedding_model=self.embedding,
                                  drop_out=dropout_p,
                                  use_bidirectional=use_bidirectional,
                                  n_layers = encoder_n_layer,
                                  rnn_style=self.encoder_rnn_style)
        if not share_encoder:
            self.encoder_encoding = EncoderRNN(vocab_size=vocab_size,
                                               embedding_size=embedding_size,
                                               hidden_size=hidden_size,
                                               use_cuda=self.use_cuda,
                                               batch_first=batch_first,
                                               embedding_model=self.embedding,
                                               drop_out=dropout_p,
                                               use_bidirectional=use_bidirectional,
                                               n_layers = encoder_n_layer,
                                               rnn_style=self.encoder_rnn_style)

        if attn_type == 'dot':
            self.attention = DotAttention(hidden_size, use_cuda=use_cuda)
        elif attn_type == 'general':
            self.attention = GeneralAttention(hidden_size, use_cuda=use_cuda)

        # VAE Attention layer:
        if self.use_vae_attention:
            self.vae_attention = VaeAttention(hidden_size, hidden_size, hidden_size, use_cuda, attn_method=vae_attention_method)
        else:
            self.vae_attention = None

        if self.use_vinilla_decoder:
            self.decoder = AttnDecoderRNN(embedding_size,
                                          hidden_size,
                                          vocab_size,
                                          encoder_layer=encoder_n_layer,
                                          attn_obj=self.attention,
                                          embedding_model=self.embedding,
                                          use_cuda=use_cuda,
                                          batch_first=batch_first,
                                          dropout_p=dropout_p,
                                          max_length=max_length,
                                          use_bidiretional_encoder=use_bidirectional,
                                          n_layers=decoder_n_layer,
                                          rnn_style=decoder_rnn_style,
                                          use_output_layer=False,
                                          use_first_encoder_hidden_state=False
                                          )
        else:
            if self.use_attention:
                self.decoder = AttnDecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  attn_obj=self.attention,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style,
                                                  use_vae_attention=self.use_vae_attention,
                                                  vae_attention=self.vae_attention)
            else:
                self.decoder = DecoderRNN_vae(embedding_size,
                                                  hidden_size,
                                                  vocab_size,
                                                  z_size,
                                                  encoder_layer=encoder_n_layer,
                                                  embedding_model=self.embedding,
                                                  use_cuda=use_cuda,
                                                  batch_first=batch_first,
                                                  dropout_p=dropout_p,
                                                  max_length=max_length,
                                                  use_bidiretional_encoder=use_bidirectional,
                                                  n_layers=decoder_n_layer,
                                                  rnn_style=decoder_rnn_style)
        if use_decoder_encoding:
            self.decoder_encoding = AttnDecoderRNN(embedding_size,
                                              hidden_size,
                                              vocab_size,
                                              encoder_layer=encoder_n_layer,
                                              attn_obj=self.attention,
                                              embedding_model=self.embedding,
                                              use_cuda=use_cuda,
                                              batch_first=batch_first,
                                              dropout_p=dropout_p,
                                              max_length=max_length,
                                              use_bidiretional_encoder=use_bidirectional,
                                              n_layers=decoder_n_layer,
                                              rnn_style=decoder_rnn_style)
        else:
            self.decoder_encoding = None

        if self.bow_loss:
            self.mlp_bow = nn.Linear(self.z_size, self.vocab_size)

        if self.vae_first_embedding:
            self.z2embedding = nn.Linear(self.z_size, self.embedding_size)

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            if use_decoder_encoding:
                self.decoder_encoding = self.decoder_encoding.cuda()
            if not share_encoder:
                self.encoder_encoding = self.encoder_encoding.cuda()
            self.prior_mu = self.prior_mu.cuda()
            self.prior_var = self.prior_var.cuda()
            self.post_mu = self.post_mu.cuda()
            self.post_var = self.post_var.cuda()
            if self.bow_loss:
                self.mlp_bow = self.mlp_bow.cuda()
            if self.vae_first_embedding:
                self.z2embedding = self.z2embedding.cuda()

        self.encoder_hidden = self.encoder.initHidden(self.batch_size)
        print('use cuda: ', self.use_cuda)
        print('bi-directional', use_bidirectional)

    def stop_vae_pre_train(self):
        self.decoder.vae_pre_train = False

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

        # if self.training:
        #   std = logvar.mul(0.5).exp_()
        #   eps = Variable(std.data.new(std.size()).normal_())
        #   return eps.mul(std).add_(mu)
        # else:
        #   return mu

    def pre_train(self):
        self.pre_training = True
    def stop_pre_train(self):
        self.pre_training = False

    def forward(self, input_idx, tgt_idx, context_lengths_src, context_lengths_tgt):
        encoder_input = input_idx
        decoder_input = tgt_idx

        # Pre-training:
        if self.pre_training:
            encoder_hs, encoder_ht = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src, sort = True)
            output_log_softmax, hidden, attns = \
                self.decoder_encoding(decoder_input, encoder_hs, torch.LongTensor(context_lengths_src),
                                      encoder_ht)
            return output_log_softmax

        # VAE encoding part:
        if self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)
        else:
            encoder_hidden = self.encoder_encoding.initHidden(self.batch_size)
            encoder_hs_src, encoder_ht_src_ori = self.encoder_encoding(encoder_input, encoder_hidden, context_lengths_src)
        encoder_ht_src = torch.chunk(encoder_ht_src_ori, self.encoder_n_layer, dim=1)[-1]  # Only consider the last layer!
        encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)

        if self.training:
            if self.use_decoder_encoding:
                output_log_softmax, hidden, attns = \
                    self.decoder_encoding(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori)
                if self._batch_first:
                    encoder_ht_tgt = hidden.transpose(0, 1)[-1]
                else:
                    encoder_ht_tgt = hidden[-1]
                encoder_ht_tgt = encoder_ht_tgt.unsqueeze(1)
            else:
                _, encoder_ht_tgt_ori = self.encoder(decoder_input, self.encoder_hidden,
                                                                  context_lengths_tgt, sort = False)
                encoder_ht_tgt = torch.chunk(encoder_ht_tgt_ori, self.encoder_n_layer, dim=1)[
                    -1]  # Only consider the last layer!
                encoder_ht_tgt = encoder_ht_tgt.contiguous().view(encoder_ht_tgt.size(0), 1, -1)


        # VAE
        if self.mean_pooling:
            mean_pooling_prior = torch.mean(encoder_hs_src, 1, True)
            z_mu_prior = self.prior_mu(mean_pooling_prior)
            z_logvar_prior = self.prior_var(mean_pooling_prior)
        else:
            z_mu_prior = self.prior_mu(encoder_ht_src)
            z_logvar_prior = self.prior_var(encoder_ht_src)
        if self.active_function != 'None':
            z_mu_prior = self.active_func(z_mu_prior)
            z_logvar_prior = self.active_func(z_logvar_prior)

        z_s = []
        if self.training:
            if self.cat_for_post:
                post_input = torch.cat([encoder_ht_tgt, encoder_ht_src], dim=2)
            else:
                post_input = encoder_ht_tgt
            if self.mean_pooling:
                mean_pooling_post = torch.mean(hidden, 1, True)
                post_input = torch.cat([mean_pooling_post, mean_pooling_prior], dim=2) if self.cat_for_post else mean_pooling_post

            z_mu_post = self.post_mu(post_input)
            z_logvar_post = self.post_var(post_input)
            if self.active_function != 'None':
                z_mu_post = self.active_func(z_mu_post)
                z_logvar_post = self.active_func(z_logvar_post)

            for i in range(self.z_sample_number):
                z = self.reparameterize(z_mu_post, z_logvar_post)
                z_s.append(z)
        else:
            for i in range(self.z_sample_number):
                z = self.reparameterize(z_mu_prior, z_logvar_prior)
                z_s.append(z)

        # VAE Decoding part:
        if not self.share_encoder:
            encoder_hs_src, encoder_ht_src_ori = self.encoder(encoder_input, self.encoder_hidden, context_lengths_src)

        output_log_softmax_s = []
        for i in range(self.z_sample_number):
            z = z_s[i]
            if self.vae_first_embedding:
                z_embedding = self.z2embedding(z)

            if self.use_vinilla_decoder:
                output_log_softmax, hidden, _ = \
                    self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src),
                                 encoder_ht_src_ori, use_z_embedding=True, z_embedding=z_embedding)
            else:
                if self.use_attention:
                    output_log_softmax, hidden, _ = \
                        self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z)
                else:
                    output_log_softmax, hidden = \
                        self.decoder(decoder_input, encoder_hs_src, torch.LongTensor(context_lengths_src), encoder_ht_src_ori, z)
            output_log_softmax_s.append(output_log_softmax)

        if (self.z_sample_number > 1):
            output_log_softmax = torch.exp(output_log_softmax_s[0])
            for i in range(0, self.z_sample_number):
                output_log_softmax = output_log_softmax + torch.exp(output_log_softmax_s[i])
            output_log_softmax/=self.z_sample_number
            output_log_softmax.log_()
        else:
            output_log_softmax = output_log_softmax_s[0]
            if self.bow_loss:
                z = z_s[0]
                bow_log_softmax =  F.log_softmax(self.mlp_bow(z))


        if self.training:
            if self.bow_loss:
                return output_log_softmax, z_mu_prior, z_logvar_prior, z_mu_post, z_logvar_post, bow_log_softmax
            else:
                return output_log_softmax, z_mu_prior, z_logvar_prior, z_mu_post, z_logvar_post
        else:
            return output_log_softmax


class cnn_gmmvae(nn.Module):
    pass

if __name__ == '__main__':


    x = torch.randn(2,4)
    y = torch.randn(2,4)
    print(x)
    print(y)
    print(torch.max(x, dim=1)[0])
    print(torch.max(x, dim=1)[1].unsqueeze(1))
    print(y[:, torch.max(x, dim=1)[1]])
    print(y[range(2), torch.max(x, dim=1)[1]])
    exit()

    def z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs):
        # E_(q(c│x, y) ) KL(q(z | x, y) | | p(z | c))
        # gaussion_mus & gaussion_vars: k x z_dim
        # mu & logvar: batch_size x 1 x z_dim
        # tgt_probs: batch_size x k
        k = gaussion_mus.size(0)
        mu_repeat = mu.repeat(1, k, 1)  # batch_size x k x z_dim
        logvar_repeat = logvar.repeat(1, k, 1)
        gaussion_logvars = gaussion_vars.pow(2).log()   # 注意标准差和方差的区别
        print(gaussion_logvars, logvar)
        print(gaussion_logvars - logvar_repeat)
        print((logvar_repeat.exp().div(gaussion_logvars.exp())).size())
        print((gaussion_logvars - logvar_repeat).size())
        print(torch.bmm((gaussion_mus - mu_repeat), ((gaussion_mus - mu_repeat) / gaussion_logvars.exp()).transpose(1, 2)).size())

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussion_logvars.exp()), dim=2)
                + torch.sum((gaussion_mus - mu_repeat).pow(2)/gaussion_logvars.exp(), dim=2)
                # + torch.sum(torch.bmm((gaussion_mus - mu_repeat), ((gaussion_mus - mu_repeat) / gaussion_logvars.exp()).transpose(1, 2)), dim=2)
                - mu.size(2)
                + torch.sum(gaussion_logvars - logvar_repeat, dim=2)
        )  # batch_size x k
        print('kl')
        print(kl)
        return torch.sum(kl * tgt_probs, dim=1)


    gaussion_mus = torch.randn(5, 2)
    gaussion_vars = torch.randn(5, 2) * 1
    mu = torch.randn(3, 1, 2)
    logvar = torch.randn(3, 1, 2)
    tgt_probs = torch.rand(3, 5)
    print(z_kl(gaussion_mus, gaussion_vars, mu, logvar, tgt_probs))
    exit()


    a = torch.rand(2, 5)
    b = torch.rand(4, 1, 5)
    print(a, b, a+b)
    exit()


    q = torch.rand(5, 4)
    q = Variable(q).cuda()
    q_repeat = q.unsqueeze(1).repeat(1, 3, 1)
    print(q_repeat)
    mus = torch.rand(3, 4)
    mus = Variable(mus).cuda()
    vars = torch.rand(3, 4)
    vars = Variable(vars).cuda()

    print(q, mus, vars)
    print(q-mus[0])
    print((q_repeat-mus)/vars)

    print(cluster_prob(q, 3, mus, vars))

    pass