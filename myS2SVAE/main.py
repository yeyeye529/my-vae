# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import sys
sys.path.insert(0, '..')

import torch
from Model import vinilla_seq2seq
from Model import vinilla_vae
from Model import vnmt
from Model import gmmvae
from utils import Utils
from Vocabulary import vocabulary
from evaluator import evaluator
from utils import my_logging
import logging
from trainer import trainer
from Batch import Batch
from evals import knn
import time

import argparse
import os


# Parameters:
def save_parameters(config_file, parameters, model_name, fn, fn_unparallel = None):
    data = {
        'parameters': parameters,
        'model_name': model_name,
        'fn': fn,
        'fn_unparallel': fn_unparallel
    }
    import json
    with open(config_file, 'w') as f:
        json.dump(data, f, indent=4)

def read_parameters(config_file = None):
    import json
    data = {
        'parameters': None,
        'model_name': None,
        'fn': None,
        'fn_unparallel': None
    }

    if config_file != None:
        with open(config_file, 'r') as f:
            data_ = json.load(f)
            if 'parameters' in data_:
                data['parameters'] = data_['parameters']
            if 'model_name' in data_:
                data['model_name'] = data_['model_name']
            if 'fn' in data_:
                data['fn'] = data_['fn']
            if 'fn_unparallel' in data_:
                data['fn_unparallel'] = data_['fn_unparallel']
    else:
        # Default parameters
        parameters = {
            'use_cuda': True,  # 是否使用GPU
            'use_unparallel': False,  # 是否混合非平行语料训练
            'unparallel_mix_rate': 0.0,   # 非平行语料混合率
            'batch_first': True,  # 第一个维度是batch吗
            'min_sent_len': 1,  # 句子长度下限
            'max_sent_len': 128,  # 句子长度上限
            'max_vocab_size': 50000,  # 词典大小上限。
            'remove_low_freq': 1,  # <= remove_low_freq, remove. 移走低频词。
            'batch_size': 64,  # Quaro: 64 MSCOCO: 128(try)
            'epoch_num': 60,  # 训练轮数
            'max_len': 128,  # 句子最大长度：读入了训练集以后这个最大长度会发生改变。根据具体数据集而言。
            'truncated': True,  # 是否截断句子。
            'truncated_len': 50,  # 截断句子的长度。
            # NN Parameters:
            'embedding_size': 128,
            'hidden_size': 256,
            'learning_rate': 1e-3,  # (***) ADAM: [1e-3], 5e-4; SGD: 1e-2
            'drop_out': 0.3,
            'verbose_freq': 100,  # 隔多少个batch输出一次信息
            'stop_threshold': 1e-3,  # 这个现在没有用。
            'lr_decay': True,  # 是否学习率衰减。
            'lr_decay_period': 6000,  # 这个暂时也没有用。目前的衰减方案是训练集上loss上升或者训练过半。
            'vocab_size': -1,  # 词典大小。读入词典后此项发生改变。
            'bi_directional': True,  # 是否在encoder上用双向rnn。
            'n_layer_encoder': 2,  # encoder层数
            'n_layer_decoder': 1,  # decoder层数
            'dev_loss_notdecrease_thres': 3,  # 暂时没用。可不管。
            'decoder_rnn_style': 'gru',  # encoder的rnn类型，可以用gru或lstm
            'encoder_rnn_style': 'gru',  # decoder的rnn类型，同上。
            'attention_style': 'dot',  # attention的类型。dot就是内积型。
            'use_output_layer': False,  # Default: False. rnn decoder是否要加一个输出层。
            'use_first_encoder_hidden_state': False,  # Default: False. 是否用双向rnn第一个时刻的反向hidden state作为decoder的初始状态。
            # Other
            'bos_idx': 0,  # begin of sentence 的词典id。
            'eos_idx': 1,  # end of sentence 的词典id。
            'unk_idx': 2,  # 未登录词 的词典id。
            'pad_idx': 3,  # <pad> 的词典id。
            # beam search
            'beam_size': 5,
            # VAE:
            'share_encoder': True,  # vae 的encoding和decoding部分是否共享encoder。
            'z_size': 2,  # (***) 隐空间维度: 建议：等于hidden_size 256
            'z_sample_num': 4,  # vae采样z的个数
            'training_z_sample_num': 1,  # 忽略
            'decoder_input_dropout': 0.0,  # encoder dropout概率，建议设成0.
            'active_func': 'None',  # 激活函数类型。不用激活函数就设成：'None'
            # Pre-training:
            'pre_training': False,  # 是否要预训练。
            'pre_training_epoch': 10,  # 预训练轮数
            # vnmt:
            'mean_pooling': False,  # (***) 是否使用mean_pooling
            'decoder_attention': True,  # decoder端是否用attention
            'vae_attention': True,  # (***) 是否用vae attention
            'vae_attention_method': 'asign_alla',   # vae attention方式： ['share', 'dot', 'general', 'asign_alla', 'asign_allz']
            'bow_loss': True,  # (***) 是否使用bow loss
            'bow_loss_w': 1.0,
            'use_decoder_encoding': False,  # 在encoding部分是否加入decoder。
            'vae_first_embedding': False,  # 是否将隐变量z作为第一个时刻的embedding使用。
            'without_post_cat': False, # 要不要将原句和目标句的隐变量拼接起来。
            # Testing:
            'filter': 'maxbleu', # ['maxbleu', 'wmd'] 筛选方式。
            # gmmvae:
            'k': 20, # !
            'cluster_check_period': 20,  # 每隔多少个epoch输出一张聚类情况图
            'batch_normalize': False,
            'gmm_period': 80,
            'gmm_init_gaussian_parameters': False,
            'cluster_loss': False,
            'closs_lambda': 0.0,
            'adjust_gaussian_var': False,
            'pretrain_s2s': False, # !
            's2s_model_fn': 'models/model_s2s_tl=50.md',
            'alternating_training': False,
            'resample_gaussian': False,
            'different_lr': False,
            'vae_first_h': True,  # !
            'hidden_reconstruction_loss': False,
            'hidden_reconstruction_loss_w': 1.0,
            'draw_pics': True,
            'pics_dir': '0_amazon_reviews_dim=2_meanZ_var=0.1(fix)_bow=1.0_klw=0.01_infoLoss', # !
            'kl_annealing_style': 'none', # ['faster', 'slower', 'none', 'fasterer'] # !
            'begin_with': 0.0, # !
            're_gaussian': False, # !
            # cls
            'use_cls': False,
            # save:
            'ckpt_period': 5000,
            'mean_zloss': True,  # !
            'KLD_weight': 0.0,
            'decoder_use_c': False, # 好像有问题！先设置成False吧！
            'eval_mode': 'normal',
            'unconditional': True,  # cvae/conditional vae
            'get_c_from_z': False,
            'sampling_period': 100, # sampling sentences from prior.
            'teacher_forcing_rate': 0.0,
            # info loss
            'info_loss': True,  # 目前好像有点问题。
            'info_loss_weigt': 1.0
        }
        model_name = 'gmmvae'
        fn_quora = {
            'trn_src': '../data/quora_duplicate_questions_trn.src',
            'trn_tgt': '../data/quora_duplicate_questions_trn.src',  # tgt
            'dev_src': '../data/quora_duplicate_questions_dev.src',
            'dev_tgt': '../data/quora_duplicate_questions_dev.tgt',
            'test_src': '../data/quora_duplicate_questions_test.src.2',  # '../data/quora_duplicate_questions_test.src'
            'test_tgt': '../data/quora_duplicate_questions_test.tgt.2'  # '../data/quora_duplicate_questions_test.tgt'
        }
        fn_amazon = {
            'trn_src': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.trn',
            'trn_tgt': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.trn',
            'dev_src': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.test',
            'dev_tgt': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.test',
            'test_src': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.test',
            'test_tgt': '/mnt/cephfs/lab/shiwenxian/dataset/amazon_reviews/laptop_reviews_sent.test',
        }
        fn_quora_unparallel = {
            'trn_src': '../data/unparallel_questions',
            'trn_tgt': '../data/unparallel_questions',
        }
        fn_mscoco = {
            'trn_src': '../data/mscoco_top3_trn.src',
            'trn_tgt': '../data/mscoco_top3_trn.tgt',
            'dev_src': '../data/mscoco_top3_dev.src',
            'dev_tgt': '../data/mscoco_top3_dev.tgt',
            'test_src': '../data/mscoco_val.src',
            'test_tgt': '../data/mscoco_val.tgt'
        }
        fn = fn_amazon
        fn_unparallel = fn_quora_unparallel
        data = {
            'parameters': parameters,
            'model_name': model_name,
            'fn': fn,
            'fn_unparallel': fn_unparallel  # fn_unparallel
        }

    # Check model name:
    # TODO: only support seq2seq/vnmt/VAE model now.
    if data['model_name'] not in ['seq2seq', 'vnmt', 'VAE', 'gmmvae']:
        raise ValueError("Error model name: " + data['model_name'])

    # print("Model:", data['model_name'])
    # print("K:", data['k'])
    return data

def preprocessing(fn, parameters, fn_unparallel, use_cuda):
    logger = logging.getLogger('preprocessing')
    # Read data and pre-processing:
    sentences_word = Utils.read_data(fn, min_str_len=parameters['min_sent_len'], max_str_len=parameters['max_sent_len'])
    vocab = vocabulary(max_vocab_size=parameters['max_vocab_size'])
    parameters['use_unparallel'] = parameters['use_unparallel'] and (fn_unparallel != None)
    if parameters['use_unparallel']:
        sentences_word_up = Utils.read_data(fn_unparallel, min_str_len=parameters['min_sent_len'],
                                            max_str_len=parameters['max_sent_len'])
        vocab.build_vocab(sentences_word['trn'][0] + sentences_word['trn'][1],
                          parameters['remove_low_freq'])
        # vocab.build_vocab(sentences_word['trn'][0] + sentences_word['trn'][1] + sentences_word_up['trn'][0], parameters['remove_low_freq'])  # <--- Here!
    else:
        vocab.build_vocab(sentences_word['trn'][0] + sentences_word['trn'][1],
                          parameters['remove_low_freq'])
    sentences_ids = {}
    sentences_lengths = {}
    max_lengths = {}
    for what in sentences_word:
        sentences_ids[what] = [[], []]
        sentences_lengths[what] = [[], []]
        sentences_ids[what][0], sentences_lengths[what][0], l1 = vocab.vocab_sent(sentences_word[what][0],
                                                                                  truncated=parameters['truncated'],
                                                                                  truncated_length=parameters[
                                                                                      'truncated_len'])
        if what == 'test':
            sentences_ids[what][1], sentences_lengths[what][1], l2 = vocab.vocab_sent_for_multi_refs(
                sentences_word[what][1],
                truncated=parameters['truncated'],
                truncated_length=parameters['truncated_len'])
        else:
            sentences_ids[what][1], sentences_lengths[what][1], l2 = vocab.vocab_sent(sentences_word[what][1],
                                                                                      truncated=parameters['truncated'],
                                                                                      truncated_length=parameters[
                                                                                          'truncated_len'])
        max_lengths[what] = max(l1, l2)
    if parameters['use_unparallel']:
        sentences_ids_up = [[], []]
        sentences_lengths_up = [[], []]
        sentences_ids_up[0], sentences_lengths_up[0], l1 = vocab.vocab_sent(sentences_word_up['trn'][0],
                                                                            truncated=parameters['truncated'],
                                                                            truncated_length=parameters[
                                                                                'truncated_len'])
        sentences_ids_up[1], sentences_lengths_up[1], l2 = vocab.vocab_sent(sentences_word_up['trn'][1],
                                                                            truncated=parameters['truncated'],
                                                                            truncated_length=parameters[
                                                                                'truncated_len'])
        # sentences_ids['trn'][0] += sentences_ids_up[0]
        # sentences_ids['trn'][1] += sentences_ids_up[1]
        # sentences_lengths['trn'][0] += sentences_lengths_up[0]
        # sentences_lengths['trn'][1] += sentences_lengths_up[1]
    else:
        sentences_ids_up = None
        sentences_lengths_up = None

    # Setting parameters:
    parameters['use_cuda'] = parameters['use_cuda'] and use_cuda and torch.cuda.is_available()
    parameters['vocab_size'] = len(vocab.vocab)
    parameters['max_len'] = min(parameters['max_sent_len'], max_lengths['trn'])

    print('Max sentences length =',parameters['max_len'])
    logger.info('Max sentences length = '+ str(parameters['max_len']))
    print('Sentences pairs in training set:', len(sentences_ids['trn'][0]))
    logger.info('Sentences pairs in training set:' + str(len(sentences_ids['trn'][0])))
    return vocab, sentences_ids, sentences_lengths, sentences_ids_up, sentences_lengths_up
    pass

def init_model(model_type, parameters):
    use_cuda = parameters['use_cuda']
    if (model_type == 'seq2seq'):
        model = vinilla_seq2seq(embedding_size = parameters['embedding_size'],
                                hidden_size = parameters['hidden_size'],
                                vocab_size = parameters['vocab_size'],
                                batch_size = parameters['batch_size'],
                                max_length=parameters['max_len'],
                                use_cuda=use_cuda,
                                batch_first=parameters['batch_first'],
                                dropout_p = parameters['drop_out'],
                                use_bidirectional=parameters['bi_directional'],
                                encoder_n_layer=parameters['n_layer_encoder'],
                                decoder_n_layer=parameters['n_layer_decoder'],
                                decoder_rnn_style=parameters['decoder_rnn_style'],
                                encoder_rnn_style=parameters['encoder_rnn_style'],
                                use_attention=parameters['decoder_attention'],
                                attn_type=parameters['attention_style'],
                                use_output_layer=parameters['use_output_layer'],
                                use_first_encoder_hidden_state=parameters['use_first_encoder_hidden_state'])
    elif (model_type == 'VAE'):
        model = vinilla_vae(embedding_size=parameters['embedding_size'],
                            hidden_size=parameters['hidden_size'],
                            vocab_size=parameters['vocab_size'],
                            batch_size=parameters['batch_size'],
                            z_size=parameters['z_size'],
                            max_length=parameters['max_len'],
                            use_cuda=use_cuda,
                            batch_first=parameters['batch_first'],
                            dropout_p=parameters['drop_out'],
                            use_bidirectional=parameters['bi_directional'],
                            encoder_n_layer=parameters['n_layer_encoder'],
                            decoder_n_layer=parameters['n_layer_decoder'],
                            decoder_rnn_style=parameters['decoder_rnn_style'],
                            encoder_rnn_style=parameters['encoder_rnn_style'],
                            share_encoder=parameters['share_encoder'],
                            decoder_input_dropout=parameters['decoder_input_dropout'],
                            vae_first_h=parameters['vae_first_h'],
                            bow_loss=parameters['bow_loss'])
    elif (model_type == 'vnmt'):
        model = vnmt(embedding_size=parameters['embedding_size'],
                     hidden_size=parameters['hidden_size'],
                     vocab_size=parameters['vocab_size'],
                     batch_size=parameters['batch_size'],
                     z_size=parameters['z_size'],
                     max_length=parameters['max_len'],
                     use_cuda=use_cuda,
                     batch_first=parameters['batch_first'],
                     dropout_p=parameters['drop_out'],
                     use_bidirectional=parameters['bi_directional'],
                     encoder_n_layer=parameters['n_layer_encoder'],
                     decoder_n_layer=parameters['n_layer_decoder'],
                     decoder_rnn_style=parameters['decoder_rnn_style'],
                     encoder_rnn_style=parameters['encoder_rnn_style'],
                     mean_pooling=parameters['mean_pooling'],
                     use_attention=parameters['decoder_attention'],
                     active_function=parameters['active_func'],
                     vae_attention=parameters['vae_attention'],
                     vae_attention_method=parameters['vae_attention_method'],
                     z_sample_num=parameters['training_z_sample_num'],
                     bow_loss=parameters['bow_loss'],
                     use_decoder_encoding=parameters['use_decoder_encoding'],
                     share_encoder=parameters['share_encoder'],
                     vae_first_h=parameters['vae_first_h'])
    elif (model_type == 'gmmvae'):
        model = gmmvae(embedding_size=parameters['embedding_size'],
                       hidden_size=parameters['hidden_size'],
                       vocab_size=parameters['vocab_size'],
                       batch_size=parameters['batch_size'],
                       z_size=parameters['z_size'],
                       k=parameters['k'],
                       max_length=parameters['max_len'],
                       use_cuda=use_cuda,
                       batch_first=parameters['batch_first'],
                       dropout_p=parameters['drop_out'],
                       use_bidirectional=parameters['bi_directional'],
                       encoder_n_layer=parameters['n_layer_encoder'],
                       decoder_n_layer=parameters['n_layer_decoder'],
                       decoder_rnn_style=parameters['decoder_rnn_style'],
                       encoder_rnn_style=parameters['encoder_rnn_style'],
                       mean_pooling=parameters['mean_pooling'],
                       use_attention=parameters['decoder_attention'],
                       active_function=parameters['active_func'],
                       vae_attention=parameters['vae_attention'],
                       vae_attention_method=parameters['vae_attention_method'],
                       z_sample_num=parameters['training_z_sample_num'],
                       # cat_for_post=parameters['cat_for_post'],
                       bow_loss=parameters['bow_loss'],
                       vae_first_embedding=parameters['vae_first_embedding'],
                       share_encoder=parameters['share_encoder'],
                       resample_gaussian=parameters['resample_gaussian'],
                       batch_normalize=parameters['batch_normalize'],
                       vae_first_h=parameters['vae_first_h'],
                       hidden_reconstruction=parameters['hidden_reconstruction_loss'],
                       decoder_use_c=parameters['decoder_use_c'],
                       get_c_from_z=parameters['get_c_from_z'],
                       unconditional=parameters['unconditional'],
                       teacher_forcing_rate=parameters['teacher_forcing_rate'],
                       bos_idx=parameters['bos_idx'],
                       eos_idx=parameters['eos_idx'])
    return model

if __name__ == '__main__':
    # todo: batch normalization

    # Argument Parsing:
    parser = argparse.ArgumentParser(description='RNNVAE')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', action='store_true', help='training mode')
    group.add_argument('-p', action="store_true", help='predicting mode')
    parser.add_argument('-c', required=False, help='config file path', type=str)
    parser.add_argument('-model', required=True, help='save(train)/load(predict) model file name.', type=str)
    parser.add_argument('--use-cuda', help='use cuda (default: False)', action="store_true", default=False)
    parser.add_argument('--gpu-idx', help='gpu index (defalt: 0). You can only choose 0-3 in 210 server.',
                        type=int, default=0)
    parser.add_argument('--save-parameters', help='save current parameter settings.',
                        type=str)
    parser.add_argument('--pretrain-model', help='pre-train model name.', type=str, default="")

    args = parser.parse_args()
    Train = args.t
    model_fn = args.model
    models_folder = os.path.join("models", args.model)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    results_folder = os.path.join("results", args.model)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    config_fn = args.c
    gpu_idx = args.gpu_idx
    use_cuda = args.use_cuda
    pretrain_model = args.pretrain_model
    print("pre-train-model:", pretrain_model)

    # Logging file:
    time_ticks = time.time()
    if not os.path.exists('loggings'):
        os.makedirs('loggings')
    if Train:
        my_logging.create_logger("loggings/" + model_fn + ".train" + "." + str(time_ticks) + ".log")
    else:
        my_logging.create_logger("loggings/" + model_fn + ".eval" + "." + str(time_ticks) + ".log")
    logger = logging.getLogger('main')

    # Read/save config file:
    data = read_parameters(config_fn)  # if c == None, use the default parameters.
    if (args.save_parameters):
        logger.info("Saving config to " + args.save_parameters)
        save_parameters(args.save_parameters, data['parameters'], data['model_name'], data['fn'], data['fn_unparallel'])
    else:
        logger.info("Saving config to " + "loggings/" + model_fn + ".config")
        if not os.path.exists("loggings/" + model_fn + ".config"):
            save_parameters("loggings/" + model_fn + "." + str(time_ticks) + ".config", data['parameters'],
                            data['model_name'], data['fn'], data['fn_unparallel'])
    parameters = data['parameters']
    model_name = data['model_name']
    fn = data['fn']
    fn_unparallel = data['fn_unparallel']

    # Pre-processing:
    vocab, sentences_ids, sentences_lengths, sentences_ids_up, sentences_lengths_up \
        = preprocessing(fn, parameters, fn_unparallel, use_cuda)

    # read class information:
    if parameters['use_cls']:
        cls_id, cls, _ = Utils.load_class_info("../data/mscoco_top3_trn.info")
        print('cls num = ', len(cls))
        parameters['k'] = len(cls)
    else:
        cls_id = cls = None

    with torch.cuda.device(gpu_idx):
        model = init_model(model_name, parameters)

        if Train:
            # Begin training:
            save_model_file_name = model_fn + ".md"
            eval_dev = evaluator(sentences_ids['dev'][0], sentences_ids['dev'][1], sentences_lengths['dev'][0],
                                 sentences_lengths['dev'][1],
                                 parameters, vocab, model_type=model_name)
            train_trn = trainer()
            batch_control = Batch(sentences_ids['trn'][0], sentences_ids['trn'][1],
                                  sentences_lengths['trn'][0], sentences_lengths['trn'][1],
                                  parameters['batch_size'], is_shuffle=True, use_cls=parameters['use_cls'],
                                  src_cls=cls_id, tgt_cls=cls_id)
            if parameters['use_unparallel']:
                batch_control.add_unparallel(sentences_ids_up[0], sentences_ids_up[1], sentences_lengths_up[0],
                                             sentences_lengths_up[1], mix_rate=parameters['unparallel_mix_rate'])

            if parameters['pretrain_s2s']:
                logging.info("Loading pretrain seq2seq model " + parameters['s2s_model_fn'])
                print("Loading pretrain seq2seq model", parameters['s2s_model_fn'])
                s2s_model = init_model("seq2seq", parameters)
                s2s_model.load_state_dict(torch.load(parameters['s2s_model_fn']))
                train_trn.load_seq2seq_parameters(s2s_model=s2s_model, my_model=model)


            train_trn.training(model, batch_control,
                               parameters, vocab,
                               model_name=model_fn,
                               save_model=True,
                               save_model_dir=models_folder,  # <-- filename
                               save_results_dir=results_folder,
                               optimizer_type='ADAM', evaluator=eval_dev, model_type=model_name,
                               pretrain_model=pretrain_model,
                               gmm_init_gaussian_parameters=parameters['gmm_init_gaussian_parameters'],
                               gmm_period=parameters['gmm_period']
                               )
        # Begin testing:
        load_model_file_name = os.path.join(models_folder, model_fn + ".pkl")  # model_fn + ".md"
        output_trans_sentences_file_name = os.path.join(results_folder, "trans_sentences_" + model_fn + ".txt")
        output_hyp_file_name = os.path.join(results_folder, "hyp_" + model_fn + ".txt")
        output_ref_file_name = os.path.join(results_folder, "ref_" + model_fn + ".txt")
        output_all_trans_file_name = os.path.join(results_folder, "all_trans_" + model_fn + ".txt")
        if not Train:
            model.load_state_dict(torch.load(load_model_file_name))

        # test for KNN:
        # knn.knn_eval(model, 1, 64, sentences_ids['dev'][0],
        #              sentences_ids['dev'][1], sentences_lengths['dev'][0],
        #              sentences_lengths['dev'][1],True)
        # exit()

        eval_test = evaluator(sentences_ids['test'][0], sentences_ids['test'][1], sentences_lengths['test'][0],
                              sentences_lengths['test'][1],
                              parameters, vocab, model_type=model_name,  # <-- model_type
                              output=True, output_path=output_trans_sentences_file_name,
                              )  # <-- remember to change the filename!
        eval_test.eval(model, False, sample_z_num=parameters['z_sample_num'],
                       output_hyp_file=output_hyp_file_name,
                       output_ref_file=output_ref_file_name,
                       output_all_trans=output_all_trans_file_name,
                       filter=parameters['filter'],
                       eval_mode=parameters['eval_mode']
                       )
