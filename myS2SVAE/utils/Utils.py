from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement
import torch
import numpy as np
from torch.autograd import Variable
import logging

import matplotlib.pyplot as plt
def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    lengths: a Tensor
    Return: torch.ByteTensor of size (batch x max_len)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def load_class_info(fn):
    f = open(fn, "r", encoding="utf-8")
    ori_info = []
    cls_id = []
    cls_name = set()
    for line in f:
        line_split = line.strip().split("\t")
        keys = line_split[::2]
        values = line_split[1::2]
        info = dict(zip(keys, values))
        ori_info.append(info)
        cls_name.add(info['supercategory'])
    cls_name = list(cls_name)
    cls_name2id = {w:i for i, w in enumerate(cls_name)}
    for info in ori_info:
        cls_id.append(cls_name2id[info['supercategory']])
    return cls_id, cls_name, ori_info


def read_data(fn_dict, min_str_len = 1, max_str_len = 128):
    logger = logging.getLogger('read_data')
    res = {}
    name = [('trn_src', 'trn_tgt'), ('dev_src', 'dev_tgt')]
    for src_fn, tgt_fn in name:
        if src_fn not in fn_dict or tgt_fn not in fn_dict:
            continue

        with open(fn_dict[src_fn], "r") as fin_src:
            with open(fn_dict[tgt_fn], "r") as fin_tgt:

                src_sents = fin_src.readlines()
                tgt_sents = fin_tgt.readlines()
                assert (len(src_sents) == len(tgt_sents)), "Sentences number does not match between %s and %s" % (src_fn, tgt_fn)
                sent_pairs = [[], []]
                for i in range(len(src_sents)):
                    src_s = src_sents[i].strip().split()
                    tgt_s = tgt_sents[i].strip().split()
                    if len(src_s) >= min_str_len and len(src_s) <= max_str_len \
                            and len(tgt_s) >= min_str_len and len(tgt_s) <= max_str_len:
                        sent_pairs[0].append(src_s)
                        sent_pairs[1].append(tgt_s)
                    else:
                        logger.info(
                            "Sentences length <%d or >%d: %s, %s" % (min_str_len, max_str_len, src_s, tgt_s))
                        # print(src_s, tgt_s)
                # sent_pairs_sorted = [[], []]
                # sent_pairs_sorted[0] = sorted(sent_pairs[0], key=lambda d: len(d), reverse=True)
                # sent_pairs_sorted[1] = sorted(sent_pairs[1], key=lambda d: len(d), reverse=True)
                res[src_fn.split('_')[0]] = sent_pairs

    # Combine for test set:
    # If you have more than one reference to one source sentence.
    name_test = [('test_src', 'test_tgt')]
    for src_fn, tgt_fn in name_test:
        if src_fn not in fn_dict or tgt_fn not in fn_dict:
            continue

        with open(fn_dict[src_fn], "r") as fin_src:
            with open(fn_dict[tgt_fn], "r") as fin_tgt:
                src_sents = fin_src.readlines()
                tgt_sents = fin_tgt.readlines()
                assert (len(src_sents) == len(tgt_sents)), "Sentences number does not match between %s and %s" % (src_fn, tgt_fn)
                sent_pairs = [[], []]
                for i in range(len(src_sents)):
                    src_sents[i] = src_sents[i].lower()
                    tgt_sents[i] = tgt_sents[i].lower()

                    if i > 0 and src_sents[i] == src_sents[i-1]:   # already read src sent
                        src_s = None
                    else:
                        src_s = src_sents[i].strip().split()
                    tgt_s = tgt_sents[i].strip().split()
                    if (src_s == None or len(src_s) >= min_str_len and len(src_s) <= max_str_len) \
                            and len(tgt_s) >= min_str_len and len(tgt_s) <= max_str_len:
                        if src_s != None:
                            sent_pairs[0].append(src_s)
                            sent_pairs[1].append([])
                        sent_pairs[1][-1].append(tgt_s)
                    else:
                        logger.info("Sentences length <%d or >%d: %s, %s" % (min_str_len, max_str_len, src_s, tgt_s))
                        # print(src_s, tgt_s)
                # sent_pairs_sorted = [[], []]
                # sent_pairs_sorted[0] = sorted(sent_pairs[0], key=lambda d: len(d), reverse=True)
                # sent_pairs_sorted[1] = sorted(sent_pairs[1], key=lambda d: len(d), reverse=True)
                res[src_fn.split('_')[0]] = sent_pairs
    return res



    for info in fn_dict:
        fn = fn_dict[info]
        sent_list = []
        if (fn != None):
            fopen = open(fn, "r", encoding="utf-8")
            for line in fopen:
                sent = line.strip().split()
                if (len(sent) < min_str_len or len(sent) > max_str_len):
                    sent_list.append(sent)


def sparse_to_matrix(sent_list, vocab_size, use_cuda = False, pad_idx = 3, use_paraphrase_knowledge = False, word2pwords = None):
    sent_list_data = sent_list.data
    sent_num = len(sent_list_data)
    matrix = np.zeros((sent_num, vocab_size))
    for sent_id, sent in enumerate(sent_list_data):
        for w_id in sent:
            if (w_id == pad_idx):
                break
            # print(sent_id, w_id)
            if w_id == 0 or w_id == 1 or w_id == 2 or w_id == 3:
                continue
            # if w_id != 0 and w_id != 1 and w_id != 2 and w_id != 3:
            matrix[sent_id][w_id] = 1
            if use_paraphrase_knowledge and w_id in word2pwords:
                for pw in word2pwords[w_id]:
                    matrix[sent_id][pw] = 1


    if use_cuda:
        return Variable(torch.FloatTensor(matrix)).cuda()
    else:
        return Variable(torch.FloatTensor(matrix))


def get_image_class(fn):
    # image_id	203564	src_id	37	tgt_id	181	category	bicycle	supercategory	vehicle
    classes = set()
    image_c = dict()
    image_id = list()
    f = open(fn, "r")
    for line in f:
        line = line.strip().split()
        id = line[1]
        if len(image_id)>0 and id == image_id[-1]:
            continue
        supercategory = line[-1]
        classes.add(supercategory)
        image_id.append(id)
        image_c[id] = supercategory

    print(classes)
    return {
        'class': list(classes),
        'image_id': image_id,
        'image_class': image_c
    }

def sequence_mean_pooling(hs, length):
    pass

def plot_embedding(embeddings, class_range, save_file_name, title=None, ):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('TSNE for z in MSCOCO')
    plt.xlabel('X')
    plt.ylabel('Y')
    legends = []
    c_seed = 1
    for cls in class_range:
        legends.append(cls)
        ax1.scatter(embeddings[class_range[cls][0]:class_range[cls][1]][:,0],
                    embeddings[class_range[cls][0]:class_range[cls][1]][:,1],
                    # cmap=plt.cm.get_cmap("jet", 10),
                    c=plt.cm.Set1(c_seed / 20)) # cmap=plt.cm.get_cmap("jet", 10), marker='o',

        c_seed+=1

    plt.legend(legends)
    # plt.show()
    fig.savefig(save_file_name)
    # plot the result
    # vis_x = vis_data[:, 0]
    # vis_y = vis_data[:, 1]
    #
    # fig = plt.figure()
    # plt.scatter(vis_x, vis_y, c=y_data, s=1, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    # plt.show()
    # fig.savefig('test.png')


    # example2:
    # import matplotlib.pyplot as plt
    #
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    #
    # plt.figure()
    # ax = plt.subplot(111)
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
    #              color=plt.cm.Set1(y[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)

def plot_representation(xys, save_file_name, mus, vars):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.style.use('ggplot')  # ?
    # ax1.set_title('TSNE for z')
    plt.xlabel('z1')
    plt.ylabel('z2')
    legends = []
    c_seed = 1
    # for c in embedding_label



    ax1.scatter(mus[:, 0], mus[:, 1], color = "teal")
    make_ellipses(vars, mus, ax1)  # draw gaussian
    ax1.scatter(xys[:, 0], xys[:, 1], color = "crimson")  # draw sample mus

    fig.savefig(save_file_name)
    plt.close(fig)


colors = ['navy'] # , 'turquoise', 'darkorange'

def make_ellipses(covariances, mus, ax):
    import matplotlib as mpl
    for i in range(len(mus)):
    # for n, color in enumerate(colors):
        # v, w = np.linalg.eigh(covariances)
        # print(v, w)
        # u = w[0] / np.linalg.norm(w[0])
        # angle = np.arctan2(u[1], u[0])
        # angle = 180 * angle / np.pi  # convert to degrees
        # v = 2. * np.sqrt(2.) * np.sqrt(v)
        # ell = mpl.patches.Ellipse(mus, v[0], v[1],
        #                           180 + angle, color=color)
        # print(mus)
        # print(covariances[:, 0])
        # print(covariances[:, 1])
        ell = mpl.patches.Ellipse(mus[i], covariances[i, 0], covariances[i, 1],
                                  0, color='turquoise')
        ell.set_zorder(0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ax.add_artist(ell)

def plot_embedding_new(embeddings, embedding_label, save_file_name, title=None, ):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.style.use('ggplot')  #?
    ax1.set_title('TSNE for z')
    plt.xlabel('X')
    plt.ylabel('Y')
    legends = []
    c_seed = 1
    # for c in embedding_label
    for i in range(len(embedding_label)):
        ax1.scatter(embeddings[i][0],
                    embeddings[i][1],
                    # cmap=plt.cm.get_cmap("jet", 10),
                    c=plt.cm.Set1(embedding_label[i] / 200))  # cmap=plt.cm.get_cmap("jet", 10), marker='o',



    # plt.legend(legends)
    # plt.show()

    fig.savefig(save_file_name)
    # plot the result
    # vis_x = vis_data[:, 0]
    # vis_y = vis_data[:, 1]
    #
    # fig = plt.figure()
    # plt.scatter(vis_x, vis_y, c=y_data, s=1, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    # plt.show()
    # fig.savefig('test.png')


    # example2:
    # import matplotlib.pyplot as plt
    #
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    #
    # plt.figure()
    # ax = plt.subplot(111)
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
    #              color=plt.cm.Set1(y[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)

def get_paraphrase_knowledge(fn, word2idx):
    f = open(fn, "r", encoding="utf-8")
    word2pwords = {}
    for line in f:
        word, pwords = line.strip().split("\t")
        pwords = pwords.split()
        if word in word2idx:
            word2pwords[word2idx[word]] = [word2idx[w] for w in pwords if w in word2idx]
    return word2pwords

if __name__ == '__main__':

    # cls_id, _ = load_class_info("../../data/mscoco_val.info")
    # print(cls_id)

    sent_list = [[1,2,3], [4,5,6]]
    embeddings = [[0.5, 0.2], [0.3, 0.4], [0.2, 0.1]]
    mus = [[0.4, 0.5], [0.5, 0.7]]
    vars = [[0.2, 0.3], [0.4, 0.5]]

    cluster = [1,2,3]
    plot_embedding_new(embeddings, cluster, "../pics/t.png")
    plot_representation(np.array(embeddings), "../pics/t.png", np.array(mus), np.array(vars))
    exit()

    print(sparse_to_matrix(sent_list, 10))
    print(sparse_to_matrix(sent_list, 10).dtype)





