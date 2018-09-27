import logging
from torch import optim
import Model
from Batch import Batch
import torch
from torch.autograd import Variable
from bleu import bleu as bleu_init   # TODO: 整合一下？
import numpy as np
from utils.Utils import plot_representation
import os
from collections import defaultdict

MAX_INT = 99999

class trainer(object):
    def __init__(self, ):
        super(trainer, self).__init__()
        self.logger = logging.getLogger("train")
        pass



    def set_gmm_lr(self, model, op_method, gmm_lr=1e-4, remain_lr = 1e-3):
        if op_method == "ADAM":

            # for key, v in model.named_parameters():
            #     print(key)

            opt = optim.Adam([
                {'params': [value for name, value in model.named_parameters() if name not in ['gaussion_mus', 'gaussion_vars',]]}, #  'h2z.weight', 'h2z.bias'
                # {'params': filter(lambda p: p.requires_grad, model.parameters())},
                {'params': model.gaussion_mus, 'lr': gmm_lr},
                {'params': model.gaussion_vars, 'lr': gmm_lr},
                #  {'params': model.h2z.parameters(), 'lr': gmm_lr}
            ], lr=remain_lr)
            return opt


    def adjust_kl_weight(self, batch_num, v = 'slower', begin_with = 0.0):
        # none:
        if v == "none":
            return 1.0

        # slower:
        if v == 'slower':
            if begin_with == 0.0:
                if batch_num < 10000:
                    return 0.001
                return min(((batch_num - 10000) // 1000) * 0.002 + 0.002, 1.0)
            else:
                return min(((batch_num) // 1000) * 0.002 + begin_with, 1.0)

        # faster:
        if v == 'faster':
            if begin_with == 0.0:
                if batch_num < 10000:
                    return 0.01
                return min(((batch_num - 10000) // 2000) * 0.01 + 0.01, 1.0)
            else:
                return min(((batch_num) // 2000) * 0.01 + begin_with, 1.0)

        # fasterer:
        if v == 'fasterer':
            if begin_with == 0.0:
                if batch_num < 10000:
                    return 0.1
                return min(((batch_num - 10000) // 2000) * 0.1 + 0.1, 1.0)
            else:
                return min(((batch_num) // 4000) * 0.1 + begin_with, 1.0)

        import math
        # return math.tanh(batch_num / 6400)
        if batch_num < 10000:
            return 0.01 # 0.000
        else:
            return 0.001
            return min(((batch_num - 10000) // 1000) * 0.001, 0.1)
        if batch_num < 1000:
            return 0
        else:
            return math.tanh(batch_num / 6400)

    def adjust_gaussian_var(self, model):
        print('adjust_gaussian_var=', model.gaussion_vars.data[0][0])
        if (model.gaussion_vars.data[0][0] <= 0.2):
            return
        else:
            model.gaussion_vars.data = model.gaussion_vars.data - 0.2

    def adjust_bow_weigt(self, epoch_num):
        # According to sun's paper.
        return 0.01
        if epoch_num < 10:
            return 0.01
        return min(0.1, 0.01 * (epoch_num - 10) + 0.01)

    def adjust_learning_rate(self, optimizer, last_lr):
        lr = last_lr * 0.5
        print('learning rate=', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5  # all decay half
        return lr

    def check_bleu(self, decoder_input, decoder_output_logsoftmax, tgt_lengths):
        topv, topi = decoder_output_logsoftmax.data.topk(1)
        topi.squeeze_(dim=2)
        return bleu_init(topi, decoder_input.data, tgt_lengths, tgt_lengths)

    def get_vecs(self, model, src_list, src_lengths, batch_size, use_cuda):
        model.eval()
        begin = 0
        end = begin + batch_size
        vecs = []
        while end < len(src_list):
            encoder_input = Variable(torch.LongTensor(src_list[begin: end]))
            batch_src_lengths = src_lengths[begin: end]
            encoder_input = encoder_input.cuda() if use_cuda else encoder_input
            begin += batch_size
            end += batch_size

            encoding_outputs = model.encoding(encoder_input, None, batch_src_lengths, None)
            representation_src = encoding_outputs['representation_src']
            # print(representation_src)
            vecs.extend(representation_src.squeeze().data.cpu().tolist())
        model.train()
        return vecs

    def test_active_cluster_num(self, model):
        pass

    def load_seq2seq_parameters(self, s2s_model, my_model):
        my_model_dict = my_model.state_dict()

        my_model_dict_paras = defaultdict(set)
        for k,v in my_model_dict.items():
            if 'decoder' in k.split('.')[0]:
                my_model_dict_paras['decoder'].add(k.split('.')[0])
            elif 'encoder' in k.split('.')[0]:
                my_model_dict_paras['encoder'].add(k.split('.')[0])
            elif 'embedding' in k.split('.')[0]:
                my_model_dict_paras['embedding'].add(k.split('.')[0])
        # my_model_dict_paras["embedding"] = "embedding"
        # print(my_model_dict_paras)

        # s2s_model.load_state_dict(torch.load(preTrain_model_file_name))

        s2s_model_dict = {}
        for k, v in s2s_model.state_dict().items():
            # print(k.split(".")[0])
            if k.split(".")[0] in ["encoder", "embedding", "decoder"]:
                for crpd_k in my_model_dict_paras[k.split(".")[0]]:
                    new_k = crpd_k + "." + '.'.join(k.split(".")[1:])
                # new_k = my_model_dict_paras[k.split(".")[0]] + "." + '.'.join(k.split(".")[1:])
                # print(new_k)
                    if my_model_dict[new_k].size() == v.size():
                        s2s_model_dict[new_k] = v
            # elif k.split(".")[0] == 'fc_mu':
            #     new_key = "prior_mu" + '.' + k.split(".")[1]
            #     pre_train_model_state_dict[new_key] = v
            # elif k.split(".")[0] == 'fc_var':
            #     new_key = "prior_var" + '.' + k.split(".")[1]
            #     pre_train_model_state_dict[new_key] = v

            # pre_train_model_state_dict = {k: v for k, v in preTrain_model.state_dict().items() if k in model_dict}

        my_model_dict.update(s2s_model_dict)
        my_model.load_state_dict(my_model_dict)

    def training(self, model, batch_control, parameters, vocab, model_name, model_type='seq2seq',
                 optimizer_type='SGD', pretrain_model = "", gmm_init_gaussian_parameters = False, gmm_period = MAX_INT,
                 criterion_type="NLLLoss", save_model=False, save_model_dir="", save_results_dir="",
                 evaluator=None, pp_knowledge = None):
        self.logger.info("Begin training.")
        print(gmm_init_gaussian_parameters)
        print(gmm_period)

        self.model_type = model_type
        use_cuda = parameters['use_cuda']
        model.train()

        # load pretrain model
        if pretrain_model != "":
            self.logger.info("Read pre-train model.")
            print("Read pre-train model.")
            model.load_state_dict(torch.load(pretrain_model + ".md", map_location=lambda storage, loc: storage))
            if model_type == 'gmmvae' and gmm_init_gaussian_parameters:
                from utils import GMM
                gmm = GMM.gmm(model.k)
                vecs = self.get_vecs(model, batch_control.p_src_list, batch_control.p_src_lengths, parameters['batch_size'], use_cuda)
                r = np.random.permutation(len(vecs))
                smp_num = int(len(vecs) * 1.0)
                vecs = np.array(vecs)[r, :][:smp_num]
                self.logger.info("Begin GMM.")
                print("Begin gmm")
                mus, vars = gmm.train(vecs)
                self.logger.info("Finished GMM.")
                print("End gmm")
                model.gaussion_mus.data = torch.FloatTensor(mus).cuda() if use_cuda else torch.FloatTensor(mus)
                model.gaussion_vars.data = torch.FloatTensor(vars).cuda() if use_cuda else torch.FloatTensor(mus)



        # Set optimizer:
        if (optimizer_type == 'SGD'):
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['learning_rate'])
        elif (optimizer_type == 'ADAM'):
            if not parameters['different_lr']:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['learning_rate'])
            else:
                optimizer = self.set_gmm_lr(model, optimizer_type, remain_lr=parameters['learning_rate'], gmm_lr=parameters['learning_rate'] * 0.5)


        # Set loss:
        if criterion_type == 'NLLLoss':
            if model_type == 'VAE' or model_type == 'vnmt':
                criterion = Model.vae_loss(parameters['batch_size'], 'NLLLoss', use_cuda=use_cuda,
                                     bow_loss=parameters['bow_loss'])
            elif model_type == 'seq2seq':
                criterion = Model.my_loss('NLLLoss', use_cuda=use_cuda, bow_loss=parameters['bow_loss'], pp_knowledge=pp_knowledge)
            elif model_type == 'gmmvae':
                criterion = Model.gmmvae_loss(parameters['batch_size'], loss_type='NLLLoss',
                                              use_cuda = use_cuda, bow_loss = parameters['bow_loss'],
                                              bow_loss_w=parameters['bow_loss_w'],
                                              cluster_loss=parameters['cluster_loss'],
                                              closs_lambda=parameters['closs_lambda'],
                                              hidden_rcs_w=parameters['hidden_reconstruction_loss_w'])

        # Begin training:
        ii = 0
        last_iter_loss = 0
        epoch_average_loss = 0
        epoch_average_losses = []
        dev_losses = []
        dev_loss_not_decrease_step = 0
        lr = parameters['learning_rate']

        for i in range(parameters['epoch_num']):
            print('Epoch', i)
            logging.info('Epoch '+str(i))
            bow_weight = self.adjust_bow_weigt(i)
            if gmm_init_gaussian_parameters and i % gmm_period == 0:
                from utils import GMM
                gmm = GMM.gmm(model.k)
                vecs = self.get_vecs(model, batch_control.p_src_list, batch_control.p_src_lengths, parameters['batch_size'], use_cuda)
                # print(vecs)
                r = np.random.permutation(len(vecs))
                smp_num = int(len(vecs) * 0.2)
                vecs = np.array(vecs)[r, :][:smp_num]
                self.logger.info("Begin GMM.")
                print("Begin gmm")
                print(vecs.shape)
                # continue
                mus, vars = gmm.train(vecs)
                self.logger.info("Finished GMM.")
                print("End gmm")

                model.gaussion_mus.data = torch.FloatTensor(mus).cuda() if use_cuda else torch.FloatTensor(mus)
                model.gaussion_vars.data = torch.FloatTensor(vars).cuda() if use_cuda else torch.FloatTensor(vars)
                model.gaussion_vars.data = torch.ones(vars.shape).cuda()  # keep ones.
                # print(model.gaussion_mus)
                # print(model.gaussion_vars)
            batch_control.init_batch(sample_unparallel=False)

            epoch_ii = 0
            epoch_average_loss = 0
            while batch_control.have_next_batch():
                ii += 1
                epoch_ii += 1

                if save_model and ii % parameters['ckpt_period'] == 0:
                    torch.save(model.state_dict(), os.path.join(save_model_dir, model_name + ".ckpt" + str(ii) + ".pkl"))

                if model_type == 'gmmvae' and parameters['re_gaussian'] and  ii == 2000:
                    # batch_output = batch_control.next_batch(fix_batch=True)
                    from utils import GMM
                    gmm = GMM.gmm(model.k)
                    vecs = self.get_vecs(model, batch_control.p_src_list, batch_control.p_src_lengths,
                                         parameters['batch_size'], use_cuda)
                    r = np.random.permutation(len(vecs))
                    smp_num = int(len(vecs) * 0.5)
                    vecs = np.array(vecs)[r, :][:smp_num]
                    self.logger.info("Begin GMM.")
                    print("Begin gmm")
                    # continue
                    mus, vars = gmm.train(vecs)
                    self.logger.info("Finished GMM.")
                    print("End gmm")

                    model.gaussion_mus.data = torch.FloatTensor(mus).cuda() if use_cuda else torch.FloatTensor(mus)
                    model.gaussion_vars.data = torch.FloatTensor(vars).cuda() if use_cuda else torch.FloatTensor(vars)

                    # gmm = GMM.gmm(model.k, means_init=model.gaussion_mus.squeeze().cpu().data.numpy())
                    # gmm = GMM.gmm(model.k, means_init=model.gaussion_mus.squeeze().cpu().data.numpy())
                    # print("Begin gmm")
                    # mus, vars = gmm.train(model_output['mu'].squeeze().cpu().data.numpy())
                    # print("End gmm")
                    # model.gaussion_mus.data = torch.FloatTensor(mus).cuda() if use_cuda else torch.FloatTensor(mus)
                    # model.gaussion_vars.data = torch.FloatTensor(vars).cuda() if use_cuda else torch.FloatTensor(vars)
                    # print(model.gaussion_vars)
                        # model.gaussion_vars.data = torch.ones(vars.shape).cuda()  # keep ones.
                batch_output = batch_control.next_batch(fix_batch=False)  # True
                # encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths = batch_control.next_batch()
                # encoder_input = Variable(encoder_input)
                # decoder_input = Variable(decoder_input)
                # encoder_input = encoder_input.cuda() if use_cuda else encoder_input
                # decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                encoder_input = batch_output[0]
                decoder_input = batch_output[1]
                batch_src_lengths = batch_output[2]
                batch_tgt_lengths = batch_output[3]

                # get model output
                if model_type == 'vnmt' or model_type == 'gmmvae':
                    model_output = model(*batch_output)
                else:
                    model_output = model(encoder_input, decoder_input, batch_src_lengths)

                # recal gaussian parameters.
                # gmm_init_gaussian_parameters
                if False and gmm_init_gaussian_parameters and  ii == 200:
                    # torch.save(model.state_dict(), save_model_path.split('.')[0]+"20000"+".md")
                    from utils import GMM
                    gmm = GMM.gmm(model.k)
                    print("Begin gmm")
                    mus, vars = gmm.train(model_output['mu'].squeeze().cpu().data.numpy())
                    print("End gmm")
                    model.gaussion_mus.data = torch.FloatTensor(mus).cuda() if use_cuda else torch.FloatTensor(mus)
                    model.gaussion_vars.data = torch.FloatTensor(vars).cuda() if use_cuda else torch.FloatTensor(vars)
                    print(model.gaussion_vars)
                    # model.gaussion_vars.data = torch.ones(vars.shape).cuda()  # keep ones.


                if model_type == 'seq2seq':
                    if not parameters['bow_loss']:
                        loss = criterion(model_output, decoder_input, batch_tgt_lengths)
                        bow_loss = 0.0
                    else:
                        loss, bow_loss = criterion(model_output[0], decoder_input, batch_tgt_lengths,
                                                   rnn_output = model_output[1], bow_loss_weight = bow_weight)
                elif model_type == 'VAE':
                    kl_weight = self.adjust_kl_weight(ii, v=parameters['kl_annealing_style'], begin_with=parameters['begin_with'])
                    criterion.KLD_weight = kl_weight
                    loss, kl_loss, bow_loss = criterion(log_softmax_output=model_output[0],
                                              target_output=decoder_input,
                                              mu=model_output[1],
                                              logvar=model_output[2],
                                              context_lengths=batch_tgt_lengths,
                                                 bow_log_softmax=model_output[3])
                    # bow_loss = 0.0
                elif model_type == 'vnmt':
                    kl_weight = self.adjust_kl_weight(ii, v=parameters['kl_annealing_style'], begin_with=parameters['begin_with'])
                    criterion.KLD_weight = kl_weight
                    loss, kl_loss, bow_loss = criterion(model_output[0], decoder_input, model_output[1],
                                                        model_output[2],
                                                        model_output[3], model_output[4], batch_tgt_lengths,
                                                        bow_log_softmax=model_output[-1])
                elif model_type == 'gmmvae':
                    kl_weight = self.adjust_kl_weight(ii, v=parameters['kl_annealing_style'], begin_with=parameters['begin_with'])
                    criterion.KLD_weight = kl_weight
                    loss, all_losses = criterion(log_softmax_output=model_output['output_log_softmax'],
                                                              target_output=decoder_input,
                                                              tgt_probs=model_output['cludis_tgt'],
                                                              src_probs=model_output['cludis_src'],
                                                              gaussion_mus=model_output['gaussion_mus'],
                                                              gaussion_vars=model_output['gaussion_vars'],
                                                              mu=model_output['mu'],
                                                              logvar=model_output['logvar'],
                                                              bow_log_softmax=model_output['bow_loss'],
                                                              context_lengths=batch_tgt_lengths,
                                                               hidden_reconstruction_loss = model_output['hidden_reconstruction_loss'],
                                                 real_tgt_cls=model_output['cludis_tgt_real'])

                optimizer.zero_grad()
                # if model_type == 'gmmvae' and parameters['alternating_training']:
                #     if ii % 100 < 95:
                #         # print("normal backward.")
                #         model.training_state = 2
                #         loss.backward()
                #     elif ii % 100 < 100:
                #         # print("kl backward.")
                #         model.training_state = 1
                #         loss_kl = zkl + ckl
                #         loss_kl.backward()
                #     else:
                #         model.training_state = 0
                #         loss_tot = loss + zkl + ckl
                #         loss_tot.backward()
                # else:
                #     loss.backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 2.0) # clip gradient
                optimizer.step()

                loss_float = float(loss.data[0])
                if (ii % parameters['verbose_freq'] == 0):
                    if model_type == 'VAE' or model_type == 'vnmt':
                        print(
                                'Batch %d, Loss %lf, bleu = %lf, KL loss = %lf, KL weight = %lf, bow loss = %lf, rnn loss = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output[0], batch_tgt_lengths), kl_loss,
                            criterion.KLD_weight, -bow_loss, loss + bow_loss - kl_loss * criterion.KLD_weight))
                        logging.info(
                                'Batch %d, Loss %lf, bleu = %lf, KL loss = %lf, KL weight = %lf, bow loss = %lf, rnn loss = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output[0], batch_tgt_lengths), kl_loss,
                            criterion.KLD_weight, -bow_loss, loss + bow_loss - kl_loss * criterion.KLD_weight))
                    elif model_type == 'gmmvae':
                        print(
                                'Batch %d, Loss %lf, bleu = %lf, cKL loss = %lf, zKL=%lf, KL weight = %lf, bow loss = %lf, rnn loss = %lf, closs = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output['output_log_softmax'], batch_tgt_lengths), all_losses['ckl_loss'], all_losses['zkl_loss'],
                            criterion.KLD_weight, all_losses['bow_loss'], all_losses['rnn_loss'], all_losses['c_loss']))
                        logging.info(
                                'Batch %d, Loss %lf, bleu = %lf, cKL loss = %lf, zKL=%lf, KL weight = %lf, bow loss = %lf, rnn loss = %lf, closs = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output['output_log_softmax'], batch_tgt_lengths), all_losses['ckl_loss'], all_losses['zkl_loss'],
                            criterion.KLD_weight, all_losses['bow_loss'], all_losses['rnn_loss'], all_losses['c_loss']))
                    else:
                        if parameters['bow_loss']:
                            model_output = model_output[0]
                        print('Batch %d, Loss %lf, bleu = %lf, bow_loss = %lf, bow_weight = %lf, rnn_loss = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output, batch_tgt_lengths), bow_loss, bow_weight, loss - bow_weight*bow_loss))
                        logging.info('Batch %d, Loss %lf, bleu = %lf, bow_loss = %lf, bow_weight = %lf, rnn_loss = %lf' % (
                            ii, loss, self.check_bleu(decoder_input, model_output, batch_tgt_lengths), bow_loss,
                            bow_weight, loss - bow_weight*bow_loss))
                last_iter_loss = loss_float
                epoch_average_loss += loss_float

                if ii % 20 == 1 and model_type == "gmmvae" and parameters['draw_pics']:
                    if not os.path.exists(os.path.join("pics", parameters['pics_dir'])):
                        os.mkdir(os.path.join("pics", parameters['pics_dir']))
                    plot_representation(model_output['mu'].squeeze(), os.path.join("pics", parameters['pics_dir'], "e" + str(ii) + ".png"), model.gaussion_mus, model.gaussion_vars)
            epoch_average_loss /= epoch_ii
            epoch_average_losses.append(epoch_average_loss)
            print("Epoch loss:", epoch_average_loss)
            logging.info("Epoch loss:%f" % epoch_average_loss)
            dev_loss = evaluator.loss(model, criterion)
            dev_losses.append(dev_loss)
            if len(dev_losses) >= 2 and dev_losses[-1] >= dev_losses[-2]:
                dev_loss_not_decrease_step += 1
            else:
                dev_loss_not_decrease_step = 0
            # if ((parameters['lr_decay'] == True and len(epoch_average_losses) >= 2
            #         and epoch_average_losses[-2] < epoch_average_losses[-1])
            #         or (parameters['lr_decay'] == True and dev_loss_not_decrease_step >= parameters['dev_loss_notdecrease_thres'])
            #         or (parameters['lr_decay'] == True and i == parameters['epoch_num'] // 2)
            #     ):  # Learning rate decay
            if (
                 # (parameters['lr_decay'] == True and len(epoch_average_losses) >= 2
                 # and epoch_average_losses[-2] < epoch_average_losses[-1])
                 # or
                 (parameters['lr_decay'] == True and i > 0 and i == parameters['epoch_num'] // 2)
            ):  # Learning rate decay
                lr = self.adjust_learning_rate(optimizer, lr)
                print("Lr =", lr)
                dev_loss_not_decrease_step = 0

            if model_type == 'gmmvae' and parameters['adjust_gaussian_var'] and i > 0 and i % 5 == 0:
                self.adjust_gaussian_var(model)
            if model_type == 'gmmvae' and parameters['resample_gaussian']:
                model.resample_gaussian()

        if save_model:
            torch.save(model.state_dict(), os.path.join(save_model_dir, model_name+".pkl"))
        return model