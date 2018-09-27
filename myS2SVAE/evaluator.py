# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
from Batch import Batch
from torch.autograd import Variable
from beam_search import Beam
from bleu_multiRef import bleu
from utils import Utils
import numpy as np
from time import time
from wmd.wmd import get_distance
import math
import logging
from collections import defaultdict
# from evals import knn


class evaluator(object):
    def __init__(self, src_list, tgt_list, src_lengths, tgt_lengths, parameters, vocab, model_type = 'seq2seq',
                 output = False, output_path = None, bleu_n = 4):
        super(evaluator, self).__init__()
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths
        self.use_cuda = parameters['use_cuda'] and torch.cuda.is_available()
        self.bos_id = parameters['bos_idx']
        self.eos_id = parameters['eos_idx']
        self.max_len = parameters['max_len']
        self.vocab = vocab
        self.beam_size = parameters['beam_size']
        self.output_fn = output_path if output else None
        self.batch_size = parameters['batch_size']
        self.model_type = model_type
        self.logger = logging.getLogger("eval")
        self.bleu_n = bleu_n
        # Beam search


    def eval(self, forward_nn, teaching_force, sample_z_num = 1,
             output_hyp_file = None, output_ref_file = None, output_all_trans = None,
             filter = 'maxbleu', one_ref = False, sample_from_prior = True):
        # self.check_clusters(forward_nn)
        # exit()
        logging.info("Begin evalution.")
        if output_hyp_file != None:
            fout_hyp = open(output_hyp_file, "w")
        if output_ref_file != None:
            fout_ref = open(output_ref_file, "w")
        if output_all_trans != None:
            fout_all_trans = open(output_all_trans, "w")

        record_cls = defaultdict(int)
        sent_cls = defaultdict(list)
        real_cls_ids = np.array([50, 62, 94, 98, 30, 15])

        forward_nn.eval()
        # Batch first!
        translations = []  # Translated sentence to source sentence,
        bleus = [] # bleus of translated sentences for all sentences in test set.
        bleus_tgt_to_src = [] # bleus of target sentences for all sentences in test set.
        bleus_max = []  # Max bleu of candidate sentences to one source sentences.
        bleus_min = []  # Min bleu fo candidate sentences to one source sentences.
        bleus_ave = []  # Average bleu of candidate sentences to one source sentences.
        wmds = []  # wmds of translated sentences (only consider "best" reference for every source sentence)
        wmd_ave = []  # Average wmds of candidate sentences to one source sentences.
        precision_collect = []
        recall_collect = []
        self.all_information = []  # All candidates with following information: bleu / wmd to src&tgt sentences, log_p.
        # Use this to learn the best combination formulation to select candidate.

        sent_length_group_ave_bleus = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        sent_clusters = defaultdict(list)
        # 0-4, 5-10, ...

        if self.model_type == 'seq2seq':
            sample_beam_search = sample_z_num
            sample_z_num = 1

        for i in range(len(self.src_list)):
            print(i)

            if one_ref:
                self.tgt_list[i] = [self.tgt_list[i]]
                self.tgt_lengths[i] = [self.tgt_lengths[i]]

            # Encoder:
            encoder_input = Variable(torch.LongTensor([self.src_list[i]]))
            if forward_nn.encoder_rnn_style == 'gru':
                encoder_hi = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_hi = encoder_hi.cuda() if self.use_cuda else encoder_hi
            elif forward_nn.encoder_rnn_style == 'lstm':
                encoder_h_0 = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_c_0 = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_h_0 = encoder_h_0.cuda() if self.use_cuda else encoder_h_0
                encoder_c_0 = encoder_c_0.cuda() if self.use_cuda else encoder_c_0
                encoder_hi = (encoder_h_0, encoder_c_0)
            if self.use_cuda:
                encoder_input = encoder_input.cuda()

            if self.model_type == 'vnmt':
                if forward_nn.share_encoder:
                    encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi, [self.src_lengths[i]], sort=True)
                else:
                    encoder_hs, encoder_ht = forward_nn.encoder_encoding(encoder_input, encoder_hi, [self.src_lengths[i]], sort=True)
            elif self.model_type == 'seq2seq':
                encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi, [self.src_lengths[i]], sort=True)
            elif self.model_type == 'VAE':
                if forward_nn.share_encoder:
                    encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi, [self.src_lengths[i]])
                else:
                    encoder_hs, encoder_ht = forward_nn.encoder2(encoder_input, encoder_hi, [self.src_lengths[i]])
            elif self.model_type == 'gmmvae':
                encoder_outputs = forward_nn.encoding(encoder_input, None, [self.src_lengths[i]], None, h0 = encoder_hi)
                encoder_hs = encoder_outputs['encoder_hs_src']
                encoder_ht = encoder_outputs['encoder_ht_src']
            if forward_nn.encoder_rnn_style == 'lstm':
                encoder_ht = encoder_ht[0].unsqueeze(0)

            # Decoder:
            encoder_ht_bak = encoder_ht
            encoder_hs_bak = encoder_hs
            if self.model_type == 'vnmt':
                encoder_ht_src = torch.chunk(encoder_ht, forward_nn.encoder_n_layer, dim=1)[-1]
                encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)
                if forward_nn.mean_pooling:
                    mean_pooling = torch.sum(encoder_hs, 1)
                    mean_pooling = (
                            mean_pooling / Variable(torch.FloatTensor([self.src_lengths[i]])).unsqueeze(
                        1).expand_as(mean_pooling).cuda()).unsqueeze(1)
                    # mean_pooling = torch.mean(encoder_hs, 1, True)
                    mu = forward_nn.prior_mu(mean_pooling)
                    logvar = forward_nn.prior_var(mean_pooling)
                else:
                    mu = forward_nn.prior_mu(encoder_ht_src)
                    logvar = forward_nn.prior_var(encoder_ht_src)
                if forward_nn.active_function != 'None':
                    mu = forward_nn.active_func(mu)
                    logvar = forward_nn.active_func(logvar)
            elif self.model_type == 'gmmvae':
                if sample_from_prior == True:
                    representation_src = encoder_outputs['representation_src']
                    cludis_src = forward_nn.cluster_prob(representation_src)
                    # print(cludis_src)
                    topv, topi = cludis_src.data.topk(1)
                    print("cls",topi[0][0])
                    # print(cludis_src.size())
                    print(cludis_src.squeeze().data.cpu().numpy()[real_cls_ids])
                    if topi[0][0] > 0.5:
                        sent_cls[topi[0][0]].append(i)

                    record_cls[topi[0][0]] += 1
                    continue
                    # print(forward_nn.gaussion_mus.size())
                    # print(forward_nn.gaussion_vars.size())
                    mu = forward_nn.gaussion_mus[topi[0][0]]
                    logvar = forward_nn.gaussion_vars[topi[0][0]].pow(2).log()
                    mu = mu.unsqueeze(0).unsqueeze(1)
                    logvar = logvar.unsqueeze(0).unsqueeze(1)
                    # print(mu.size())
                    # print(logvar.size())
                else:
                    # print("else")
                    mu, logvar = forward_nn.get_gaussian_parameters(encoder_outputs)
                # print(mu)
                # representation_src = encoder_outputs['representation_src']
                # print(representation_src)
                # cludis_src = forward_nn.cluster_prob(representation_src)
                # print(cludis_src)
                # topv, topi = cludis_src.data.topk(1)
                # sent_clusters[topi.cpu().numpy()[0][0]].append(i)
                # print(topv, topi)
                # print(forward_nn.gaussion_mus)
                # print(forward_nn.gaussion_vars)
                # print(forward_nn.gaussion_mus[170])
                # print(forward_nn.gaussion_vars[170])
                # exit()

            bleus_to_src = []
            bleus_to_tgt = []
            translation_to_src = []
            wmd_to_src = []
            wmd_to_tgt = []
            bleus_tgt_to_src.append(bleu([self.src_list[i][:self.src_lengths[i]]], [self.tgt_list[i]], [self.tgt_lengths[i]], n=self.bleu_n))
            if output_all_trans:
                fout_all_trans.write(' '.join([self.vocab.vocab[w] for w in self.src_list[i][:self.src_lengths[i]]])+"\n")

            info = []
            for k in range(sample_z_num):
                beam = Beam(self.beam_size, self.vocab.word2idx)
                decoded_ids = [self.bos_id]

                if self.model_type == 'VAE':
                    z = Variable(torch.zeros(1, forward_nn.z_size).normal_())
                    z = z.cuda() if self.use_cuda else z
                    if z.dim() == 2:
                        z = z.unsqueeze(0)
                if self.model_type == 'vnmt' or self.model_type == 'gmmvae':
                    std = logvar.mul(0.5).exp_()
                    eps = Variable(std.data.new(std.size()).normal_())
                    z = eps.mul(std).add_(mu)
                    # z = mu
                    z = z.cuda() if self.use_cuda else z
                    if z.dim() == 2:
                        z = z.unsqueeze(0)
                    if forward_nn.vae_first_embedding:
                        z_embedding = forward_nn.z2embedding(z)

                encoder_ht = encoder_ht_bak
                encoder_hs = encoder_hs_bak
                encoder_lengths = torch.LongTensor([self.src_lengths[i]])
                decoder_input = Variable(torch.LongTensor([[self.bos_id]]))  # input: <s>
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
                if (self.model_type == 'vnmt' or self.model_type == 'VAE' or self.model_type == 'gmmvae') and not forward_nn.share_encoder:
                    encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi,
                                                                [self.src_lengths[i]], sort=True)

                for di in range(self.max_len - 1):
                    if self.model_type == 'VAE':
                        output_log_softmax, hidden, attns = \
                            forward_nn.decoder2(decoder_input, encoder_hs, encoder_lengths, encoder_ht, z, padding = False)
                    elif self.model_type == 'vnmt' or self.model_type == 'gmmvae':
                        first_step = (di == 0)
                        if forward_nn.vae_first_embedding:
                            if di == 0:
                                use_z_embedding = True
                            else:
                                use_z_embedding = False
                            output_log_softmax, hidden, _ = \
                                forward_nn.decoder(decoder_input, encoder_hs, encoder_lengths,
                                             encoder_ht, use_z_embedding=use_z_embedding, z_embedding=z_embedding, padding=False)
                        else:
                            if forward_nn.use_attention:
                                output_log_softmax, hidden, attns = \
                                    forward_nn.decoder(decoder_input, encoder_hs, encoder_lengths, encoder_ht, z,
                                                        padding=False, first_step=first_step)
                            else:
                                output_log_softmax, hidden = \
                                    forward_nn.decoder(decoder_input, encoder_hs, encoder_lengths, encoder_ht, z,
                                                       padding=False)
                    else:
                        output_log_softmax, hidden, attns = \
                            forward_nn.decoder(decoder_input, encoder_hs, encoder_lengths, encoder_ht, padding=False)

                    if (beam.advance(output_log_softmax.squeeze(1).data) == True):   # Done
                        break

                    decoder_input = Variable(beam.get_current_state().unsqueeze(1))   # LongTensor of size [beam_search]  -> batch(beam_size) x str_len(=1)
                    if (hidden.size(0) == 1):  # if the beginning: 1 * 1 * hidden_size
                        encoder_ht = hidden.repeat(self.beam_size, 1, 1) # 1 x 1 x hidden_size -> beam_size x 1 x hidden_size
                        encoder_hs = encoder_hs.repeat(self.beam_size, 1, 1) # 1 x strlen x hidden_size -> beam_size x strlen x hidden_size
                        encoder_lengths = encoder_lengths.repeat(self.beam_size) # 1 -> beam_size
                    else:
                        encoder_ht = hidden[beam.get_current_origin()]

                    if (self.model_type == 'VAE' or self.model_type == 'vnmt' or self.model_type == 'gmmvae') and di == 0:
                        if z.dim() == 2:
                            z = z.repeat(self.beam_size, 1)
                        elif z.dim() == 3:
                            z = z.repeat(self.beam_size, 1, 1)

                if self.model_type == 'seq2seq':
                    info = []
                    log_ps = beam.get_best(sample_beam_search)[0]
                    for topk in range(sample_beam_search):
                        one_trans_info = {'bleu_to_src': -1, 'bleu_to_tgt': -1, 'wmd_to_src': -1, 'wmd_to_tgt': -1,
                                          'log_p': -1}
                        one_trans_info['log_p'] = log_ps[topk]

                        sent_seq2seq = [self.bos_id] + (beam.get_hyp(topk))

                        translation_to_src.append(sent_seq2seq)
                        for idx in range(0, len(sent_seq2seq)):
                            if sent_seq2seq[idx] == self.eos_id:
                                break
                        # for idx in range(0, len(sent_seq2seq))[::-1]:
                        #     if sent_seq2seq[idx] != self.eos_id:
                        #         break
                        sent_seq2seq = sent_seq2seq[:idx + 1]  # +2
                        if output_all_trans:
                            fout_all_trans.write(str(topk) + "\t")
                            fout_all_trans.write(' '.join([self.vocab.vocab[w] for w in sent_seq2seq]))
                            fout_all_trans.write("\n")

                        # bleu to src sent
                        bleu_value = bleu([sent_seq2seq], [[self.src_list[i]]], [[self.src_lengths[i]]], n=self.bleu_n)
                        bleus_to_src.append(bleu_value)
                        one_trans_info['bleu_to_src'] = bleu_value
                        # bleu to tgt sent
                        bleu_value = bleu([sent_seq2seq], [self.tgt_list[i]], [self.tgt_lengths[i]], n=self.bleu_n)
                        bleus_to_tgt.append(bleu_value)
                        bleus_ave.append(bleu_value)
                        one_trans_info['bleu_to_tgt'] = bleu_value
                        # WMD:
                        wmd = get_distance(' '.join([self.vocab.vocab[w] for w in sent_seq2seq]),
                                           ' '.join([self.vocab.vocab[w] for w in self.src_list[i]]))
                        if math.isnan(wmd):
                            wmd = 0
                        wmd_to_src.append(wmd)
                        one_trans_info['wmd_to_src'] = wmd
                        wmd = get_distance(' '.join([self.vocab.vocab[w] for w in sent_seq2seq]),
                                           ' '.join([self.vocab.vocab[w] for w in self.tgt_list[i][0]]))
                        if math.isnan(wmd):
                            wmd = 0
                        wmd_to_tgt.append(wmd)
                        wmd_ave.append(wmd)
                        one_trans_info['wmd_to_tgt'] = wmd
                        info.append(one_trans_info)
                    if len(info) > 0:
                        self.all_information.append(info)
                else:
                    one_trans_info = {'bleu_to_src': -1, 'bleu_to_tgt': -1, 'wmd_to_src': -1, 'wmd_to_tgt': -1,
                                      'log_p': -1}
                    sent = beam.get_hyp(0)
                    log_p = beam.get_best()[0]
                    one_trans_info['log_p'] = log_p[0]

                    decoded_ids.extend(sent)
                    for idx in range(0, len(decoded_ids)):
                        if decoded_ids[idx] == self.eos_id:
                            break
                    # for idx in range(0, len(decoded_ids))[::-1]:
                    #     if decoded_ids[idx] != self.eos_id:
                    #         break
                    decoded_ids = decoded_ids[:idx+1] # +2

                    translation_to_src.append(decoded_ids)
                    if output_all_trans:
                        fout_all_trans.write(str(k) + "\t")
                        fout_all_trans.write(' '.join([self.vocab.vocab[w] for w in decoded_ids]))
                        fout_all_trans.write("\n")
                    # bleu to src sent
                    bleu_value = bleu([decoded_ids], [[self.src_list[i]]], [[self.src_lengths[i]]], n=self.bleu_n)
                    bleus_to_src.append(bleu_value)
                    one_trans_info['bleu_to_src'] = bleu_value
                    # bleu to tgt sent
                    bleu_value = bleu([decoded_ids], [self.tgt_list[i]], [self.tgt_lengths[i]],n=self.bleu_n)
                    bleus_to_tgt.append(bleu_value)
                    one_trans_info['bleu_to_tgt'] = bleu_value
                    bleus_ave.append(bleu_value)
                    wmd = get_distance(' '.join([self.vocab.vocab[w] for w in decoded_ids]),
                                       ' '.join([self.vocab.vocab[w] for w in self.src_list[i]]))
                    if math.isnan(wmd):
                        wmd = 0
                    wmd_to_src.append(wmd)
                    one_trans_info['wmd_to_src'] = wmd
                    wmd = get_distance(' '.join([self.vocab.vocab[w] for w in decoded_ids]),
                                       ' '.join([self.vocab.vocab[w] for w in self.tgt_list[i][0]]))
                    if math.isnan(wmd):
                        wmd = 0
                    wmd_to_tgt.append(wmd)
                    wmd_ave.append(wmd)
                    one_trans_info['wmd_to_tgt'] = wmd
                    info.append(one_trans_info)

            if self.model_type == 'vnmt' or self.model_type == 'gmmvae':
                self.all_information.append(info)
            if filter == 'maxbleu':
                max_bleu_idx = bleus_to_src.index(max(bleus_to_src))  # Get the paraphrasing sentence with max bleu with [src] sentence.
            elif filter == 'wmd':
                max_bleu_idx = wmd_to_src.index(min(wmd_to_src))   # Attention!! Min distance -> more similar
            bleus.append(bleus_to_tgt[max_bleu_idx])  # Remember! Here should be the bleu to the [tgt] sentences.
            wmds.append(wmd_to_tgt[max_bleu_idx])

            if ((self.src_lengths[i] // 5) < len(sent_length_group_ave_bleus)):
                sent_length_group_ave_bleus[(self.src_lengths[i] // 5)].append(bleus_to_tgt[max_bleu_idx])

            translations.append((translation_to_src[max_bleu_idx], bleus_to_tgt[max_bleu_idx]))
            if output_hyp_file:
                fout_hyp.write(' '.join([self.vocab.vocab[w] for w in translation_to_src[max_bleu_idx]]) + "\n")
            if output_ref_file:
                fout_ref.write(' '.join([self.vocab.vocab[w] for w in self.tgt_list[i][0][:self.tgt_lengths[i][0]]]) + "\n")

            bleus_max.append(max(bleus_to_tgt))
            bleus_min.append(min(bleus_to_tgt))

            precision = self.precision(self.tgt_list[i], translation_to_src, self.tgt_lengths[i])
            recall = self.recall(self.tgt_list[i], translation_to_src, self.tgt_lengths[i])
            precision_collect.append(precision)
            recall_collect.append(recall)

        print(record_cls)
        tout = open("testout_sentcls.txt", "w")
        for cls in sent_cls:
            tout.write(str(cls) + "\n")
            for s in sent_cls[cls]:
                tout.write(' '.join([self.vocab.vocab[x] for x in self.src_list[s][:self.src_lengths[s]]]))
                tout.write("\n")
        exit()

        print("Average bleu(tgt to src):", sum(bleus_tgt_to_src) / len(bleus_tgt_to_src))
        print('Average bleu:', sum(bleus_ave) / len(bleus_ave))
        print("Average bleu(after filtering):", sum(bleus) / len(bleus))
        print('Max bleu:', sum(bleus_max) / len(bleus_max))
        print('Min bleu:', sum(bleus_min) / len(bleus_min))
        print("Precision:", sum(precision_collect) / len(precision_collect))
        print("Recall:", sum(recall_collect) / len(recall_collect))
        print("Average wmd:", sum(wmd_ave) / len(wmd_ave))
        print("Average wmd (after filtering)", sum(wmds) / len(wmds))

        logging.info("Average bleu(tgt to src): %.6f" % (sum(bleus_tgt_to_src) / len(bleus_tgt_to_src)))
        logging.info('Average bleu: %.6f' % (sum(bleus_ave) / len(bleus_ave)))
        logging.info("Average bleu(after filtering): %.6f" % (sum(bleus) / len(bleus)))
        logging.info('Max bleu: %.6f' % (sum(bleus_max) / len(bleus_max)))
        logging.info('Min bleu: %.6f' % (sum(bleus_min) / len(bleus_min)))
        logging.info("Precision: %.6f" % (sum(precision_collect) / len(precision_collect)))
        logging.info("Recall: %.6f" % (sum(recall_collect) / len(recall_collect)))
        logging.info("Average wmd: %.6f" % (sum(wmd_ave) / len(wmd_ave)))
        logging.info("Average wmd (after filtering) %.6f" % (sum(wmds) / len(wmds)))

        for i, lst in enumerate(sent_length_group_ave_bleus):
            if (len(lst) > 0):
                print(i, sum(lst) / len(lst), len(lst))
                logging.info("len=%d, ave_bleu=%.6f, sen_num=%d" % (i, sum(lst) / len(lst), len(lst)))

        if self.output_fn != None:
            f = open(self.output_fn, "w")
            for id, (trans, bv) in enumerate(translations):
                # idx = id // sample_z_num
                idx = id
                f.write(' '.join([self.vocab.vocab[x] for x in self.src_list[idx][:self.src_lengths[idx]]]))
                f.write("\t")
                f.write(' '.join([self.vocab.vocab[i] for i in trans]))
                f.write('\t' + str(bv))
                f.write("\n")
        # print(sent_clusters)

        forward_nn.train()

    def loss(self, forward_nn, criterion):
        forward_nn.eval()
        batch_control = Batch(self.src_list, self.tgt_list, self.src_lengths, self.tgt_lengths, self.batch_size,
                              is_shuffle=False)  # Shuffle = True
        losses = []
        batch_control.init_batch()
        while batch_control.have_next_batch():
            batch_output = batch_control.next_batch()
            # encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths = batch_control.next_batch()
            # encoder_input = Variable(encoder_input)
            # decoder_input = Variable(decoder_input)
            # encoder_input = encoder_input.cuda() if self.use_cuda else encoder_input
            # decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
            encoder_input = batch_output[0]
            decoder_input = batch_output[1]
            batch_src_lengths = batch_output[2]
            batch_tgt_lengths = batch_output[3]

            if self.model_type == 'vnmt':
                model_output = forward_nn(encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths)
                loss, kl_loss = criterion(model_output, decoder_input, is_train = False, context_lengths = batch_tgt_lengths)
            elif self.model_type == 'seq2seq':
                model_output = forward_nn(encoder_input, decoder_input, batch_src_lengths)
                loss = criterion(model_output, decoder_input, batch_tgt_lengths, is_train = False)
            elif self.model_type == 'VAE':
                model_output = forward_nn(encoder_input, decoder_input, batch_src_lengths)
                loss, kl_loss = criterion(model_output, decoder_input, is_train=False,
                                          context_lengths=batch_tgt_lengths)
            elif self.model_type == 'gmmvae':
                model_output = forward_nn(encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths)
                loss, ckl,zkl, bow_loss = criterion(log_softmax_output=model_output['output_log_softmax'],
                                                  target_output=decoder_input,
                                                  tgt_probs=model_output['cludis_tgt'],
                                                  src_probs=model_output['cludis_src'],
                                                  gaussion_mus=model_output['gaussion_mus'],
                                                  gaussion_vars=model_output['gaussion_vars'],
                                                  mu=model_output['mu'],
                                                  logvar=model_output['logvar'],
                                                  bow_log_softmax=model_output['bow_loss'],
                                                  context_lengths=batch_tgt_lengths,
                                                    is_train=False
                                                    )


            loss_float = float(loss.data[0])
            losses.append(loss_float)
        ave_loss = sum(losses) / len(losses)
        print("Ave loss:", ave_loss)
        logging.info("Ave loss:%f" % ave_loss)
        forward_nn.train()


        return ave_loss
        # print("Epoch", epoch_average_losses)

    def sort(self, inverse=False):
        if inverse:
            p = np.argsort(-self.src_lengths)
        else:
            p = np.argsort(self.src_lengths)
        self.src_list = self.src_list[p, :]
        self.tgt_list = self.tgt_list[p, :]
        self.src_lengths = self.src_lengths[p]
        self.tgt_lengths = self.tgt_lengths[p]

    def precision(self, references, candidates, references_lenghs):
        max_bleu_value_sum = 0
        for c in candidates:
            max_bleu_value = 0
            for i, r in enumerate(references):
                bleu_value = bleu([c], [[r]], [[references_lenghs[i]]], n=self.bleu_n)
                max_bleu_value = max(max_bleu_value, bleu_value)
            max_bleu_value_sum += max_bleu_value
        return max_bleu_value_sum / len(candidates)

    def recall(self, references, candidates, references_lenghs):
        max_bleu_value_sum = 0
        for i, r in enumerate(references):
            max_bleu_value = 0
            for c in candidates:
                bleu_value = bleu([c], [[r]], [[references_lenghs[i]]], n=self.bleu_n)
                max_bleu_value = max(max_bleu_value, bleu_value)
            max_bleu_value_sum += max_bleu_value
        return max_bleu_value_sum / len(references)

    def train_weight(self):
        alpha_range = [a/10 for a in range(11)]  # For bleu
        beta_range = [a / 10 for a in range(11)]  # For MWD
        gamma_range = [a / 10 for a in range(11)] # For log_p
        results = []
        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    com = (alpha, beta, gamma)
                    bleus = []
                    for i in range(len(self.all_information)):
                        tmp = []
                        for sample in self.all_information[i]:
                            v = sample['bleu_to_src'] * alpha + (1 - sample['wmd_to_src']) * beta + sample['log_p'] * gamma
                            bleu_to_tgt = sample['bleu_to_tgt']
                            tmp.append((v, bleu_to_tgt))
                        tmp_sorted = sorted(tmp, key=lambda d: d[0], reverse=True)
                        bleus.append(tmp_sorted[0][1])
                    results.append((com, sum(bleus) / len(bleus)))
        results_sorted = sorted(results, key=lambda d: d[1], reverse=True)
        for com, value in results_sorted:
            print(com, value)
        return results_sorted[0][0]

    def test_weight(self, weight_com):
        alpha = weight_com[0]
        beta = weight_com[1]
        gamma = weight_com[2]

        bleus = []
        for i in range(len(self.all_information)):
            tmp = []
            for sample in self.all_information[i]:
                v = sample['bleu_to_src'] * alpha + (1 - sample['wmd_to_src']) * beta + sample['log_p'] * gamma
                bleu_to_tgt = sample['bleu_to_tgt']
                tmp.append((v, bleu_to_tgt))
            tmp_sorted = sorted(tmp, key=lambda d: d[0], reverse=True)
            bleus.append(tmp_sorted[0][1])
        return sum(bleus) / len(bleus)

    def check_clusters(self,forward_nn):
        if self.model_type != "gmmvae":
            return
        mu = forward_nn.gaussion_mus[1]
        std = forward_nn.gaussion_vars[1].abs()
        print(mu, std)
        mu = forward_nn.gaussion_mus[2]
        std = forward_nn.gaussion_vars[2].abs()
        print(mu, std)
        # exit()
        forward_nn.eval()
        begin = 0
        end = begin + self.batch_size
        vecs = []
        clabels = []
        while end<len(self.src_list):
            encoder_input = Variable(torch.LongTensor(self.src_list[begin: end]))
            batch_src_lengths = self.src_lengths[begin: end]
            encoder_input = encoder_input.cuda() if self.use_cuda else encoder_input
            begin += self.batch_size
            end += self.batch_size

            encoding_outputs = forward_nn.encoding(encoder_input, None, batch_src_lengths, None)
            representation_src = encoding_outputs['representation_src']
            cludis_src = forward_nn.cluster_prob(representation_src)
            print(cludis_src.data.cpu()[0].tolist())
            topv, topi = cludis_src.data.topk(1)
            clabels.extend(topi.squeeze().cpu().tolist())
            vecs.extend(representation_src.squeeze().data.tolist())

            # sent_clusters[topi.cpu().numpy()[0][0]].append(i)

        print(len(vecs))
        print(len(clabels))
        print(clabels)
        return

        print("Begin tsne")
        smp_num = 1000
        print("sample num = ", smp_num)  # todo 搞成参数传进来
        # smp_num = 1000
        r = np.random.permutation(len(vecs))
        # smp_labels = np.array(clabels)[r][:smp_num]
        # print(smp_labels)
        # print(smp_labels)
        from utils import GMM
        gmm = GMM.gmm(K=200)
        mus, vars = gmm.train(vecs[:2000], sent_list=self.src_list,
                              vocab=self.vocab.vocab,output_class_path = "cluster_ht.txt")  # ,
        exit()
        sample_cluster = []
        mu = forward_nn.gaussion_mus[15]
        std = forward_nn.gaussion_vars[15].abs()
        for i in range(2000):
            eps = Variable(std.data.new(std.size()).normal_())
            s = eps.mul(std).add_(mu)
            sample_cluster.append(s.data.cpu().tolist())

        time0 = time()
        # print(np.array(vecs)[r, :][:500])
        # print(np.array(sample_cluster))

        mus, vars = gmm.train(vecs,  sent_list = self.src_list, vocab = self.vocab.vocab) # output_class_path = "cluster.txt",
        forward_nn.gaussion_mus.data = torch.FloatTensor(mus).cuda() if self.use_cuda else torch.FloatTensor(mus)
        forward_nn.gaussion_vars.data = torch.FloatTensor(vars).cuda() if self.use_cuda else torch.FloatTensor(mus)

        begin = 0
        end = begin + self.batch_size
        vecs = []
        clabels = []
        while end < len(self.src_list):
            encoder_input = Variable(torch.LongTensor(self.src_list[begin: end]))
            batch_src_lengths = self.src_lengths[begin: end]
            encoder_input = encoder_input.cuda() if self.use_cuda else encoder_input
            begin += self.batch_size
            end += self.batch_size

            encoding_outputs = forward_nn.encoding(encoder_input, None, batch_src_lengths, None)
            representation_src = encoding_outputs['representation_src']
            cludis_src = forward_nn.cluster_prob(representation_src)
            print(cludis_src.data.cpu()[0].tolist())
            topv, topi = cludis_src.data.topk(1)
            clabels.extend(topi.squeeze().cpu().tolist())
            vecs.extend(representation_src.squeeze().data.tolist())



        exit()

        from sklearn.manifold import TSNE
        vecs = np.array(vecs)[r, :][:smp_num]
        vecs = np.concatenate((vecs, np.array(sample_cluster)), axis=0)
        X_embedded = TSNE(n_components=2, init='pca', perplexity=30).fit_transform(vecs)  # , init='pca')
        print(X_embedded.shape)
        print("Time:%fs" % (time() - time0))
        print("Plotting")
        smp_labels = np.concatenate((smp_labels, np.array([1] * 2000)))
        Utils.plot_embedding_new(X_embedded, smp_labels, "test_50_pca_k=20.png")

        # print("Begin tsne")
        # time0 = time()
        # X_embedded = TSNE(n_components=2, perplexity=50).fit_transform(X)  # , init='pca')
        # print("Time:%fs" % (time() - time0))
        # print("Plotting")
        # Utils.plot_embedding(X_embedded, class_range, "test_p=50.png")

        exit()

    def classification(self, test_info_fn, forward_nn):
        image_class_info = Utils.get_image_class(test_info_fn)

        forward_nn.eval()
        imgVecs = {}
        for c in image_class_info['class']:
            imgVecs[c] = []

        for i in range(len(self.src_list)):
            if i > 0 and image_class_info['image_id'][i] == image_class_info['image_id'][i - 1]:
                # print(i, image_class_info['image_id'][i])
                continue
            # Encoder:
            encoder_input = Variable(torch.LongTensor([self.src_list[i]]))
            if forward_nn.encoder_rnn_style == 'gru':
                encoder_hi = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_hi = encoder_hi.cuda() if self.use_cuda else encoder_hi
            elif forward_nn.encoder_rnn_style == 'lstm':
                encoder_h_0 = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_c_0 = Variable(
                    torch.zeros(forward_nn.direction_num * forward_nn.encoder_n_layer, 1, forward_nn.hidden_size))
                encoder_h_0 = encoder_h_0.cuda() if self.use_cuda else encoder_h_0
                encoder_c_0 = encoder_c_0.cuda() if self.use_cuda else encoder_c_0
                encoder_hi = (encoder_h_0, encoder_c_0)
            if self.use_cuda:
                encoder_input = encoder_input.cuda()
            if self.model_type == 'seq2seq' or self.model_type == 'vnmt' or self.model_type == 'my_vae1' or self.model_type == 'my_vae2':
                encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi, [self.src_lengths[i]])
            elif self.model_type == 'VAE':
                if forward_nn.share_encoder:
                    encoder_hs, encoder_ht = forward_nn.encoder(encoder_input, encoder_hi, [self.src_lengths[i]])
                else:
                    encoder_hs, encoder_ht = forward_nn.encoder2(encoder_input, encoder_hi, [self.src_lengths[i]])
            if forward_nn.encoder_rnn_style == 'lstm':
                encoder_ht = encoder_ht[0].unsqueeze(0)

            if self.model_type == 'vnmt':
                encoder_ht_src = torch.chunk(encoder_ht, forward_nn.encoder_n_layer, dim=1)[-1]
                encoder_ht_src = encoder_ht_src.contiguous().view(encoder_ht_src.size(0), 1, -1)
                if forward_nn.mean_pooling:
                    mean_pooling = torch.mean(encoder_hs, 1, True)
                    mu = forward_nn.prior_mu(mean_pooling)
                else:
                    mu = forward_nn.prior_mu(encoder_ht_src)
                if forward_nn.active_function != 'None':
                    mu = forward_nn.active_func(mu)
            if self.model_type == 'my_vae1':
                matrix_src = Utils.sparse_to_matrix(encoder_input, forward_nn.vocab_size, self.use_cuda)
                z_src = forward_nn.relu(forward_nn.fc2(forward_nn.relu(forward_nn.fc1(matrix_src))))
                mu = forward_nn.relu(forward_nn.prior_mu(z_src))
            if self.model_type == 'my_vae2':
                o = forward_nn.mlp_vae(encoder_input)
                mu = o['mu']

            if image_class_info['image_class'][image_class_info['image_id'][i]] != "":
                imgVecs[image_class_info['image_class'][image_class_info['image_id'][i]]].append(mu)

        X = np.array(0)
        class_range = {}
        for c in imgVecs:
            if c == 'none':
                continue
            print(c, len(imgVecs[c]))
            matrix_np = torch.cat(imgVecs[c], dim=0).squeeze().data.numpy()
            matrix_np = matrix_np[:500]
            print(matrix_np.shape)
            if X.ndim == 0:
                X = matrix_np
                class_range[c] = [0, matrix_np.shape[0]]
            else:
                X = np.concatenate((X, matrix_np), axis=0)
                class_range[c] = [X.shape[0] - matrix_np.shape[0], X.shape[0]]

        print(class_range)
        print(X.shape)

        print("Begin tsne")
        time0 = time()
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(X)  # , init='pca')
        print(X_embedded.shape)
        print("Time:%fs" % (time() - time0))
        print("Plotting")
        Utils.plot_embedding(X_embedded, class_range, "test_50_pca.png")

        print("Begin tsne")
        time0 = time()
        X_embedded = TSNE(n_components=2, perplexity=50).fit_transform(X)  # , init='pca')
        print("Time:%fs" % (time() - time0))
        print("Plotting")
        Utils.plot_embedding(X_embedded, class_range, "test_p=50.png")

        return





