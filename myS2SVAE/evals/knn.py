# import os
# print(os.getcwd())
# os.chdir("../..")
# print(os.getcwd())

import torch
from torch.autograd import Variable
import numpy as np
# from myS2SVAE.chinese_exp import read_parameters
# from myS2SVAE.chinese_exp import preprocessing
# from myS2SVAE.chinese_exp import init_model

def get_vecs(model, src_list, src_lengths, batch_size, use_cuda):
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


def knn_eval(model, k, batch_size, src_list, tgt_list, src_lengths, tgt_lengths, use_cuda):
    model.eval()
    begin = 0
    end = begin + batch_size
    vecs = []
    tot_maps = []
    while end < len(src_list):
        print(begin)
        encoder_input = Variable(torch.LongTensor(src_list[begin: end]))
        batch_src_lengths = src_lengths[begin: end]
        # print(batch_src_lengths)
        encoder_input = encoder_input.cuda() if use_cuda else encoder_input

        decoder_input = Variable(torch.LongTensor(tgt_list[begin: end]))# .squeeze()
        batch_tgt_lengths = tgt_lengths[begin: end]
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        batch_tgt_lengths_np = np.array(batch_tgt_lengths)# .transpose()[0]

        # model.encoding(encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths_np,
        #                                   cal_tgt = True)
        encoding_outputs = model.encoding(encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths_np,
                                          cal_tgt=True)


        representation_src = encoding_outputs['representation_src'].squeeze()
        representation_tgt = encoding_outputs['representation_tgt'].squeeze()


        # print(representation_src.size())

        # representation_all = torch.cat((representation_src, representation_tgt), 1)

        # inner_product = torch.matmul(representation_all, representation_all.transpose(0, 1))
        # l_all = torch.sum(representation_all * representation_all, dim=1)
        # distance_square =  - 2 * inner_product + l_all.repeat(l_all.size(0), 1).transpose(0,1) + l_all.repeat(l_all.size(0), 1)
        # print(representation_all.size())
        # exit()
        #
        inner_product = torch.matmul(representation_src, representation_tgt.transpose(0,1))

        l_src = torch.sum(representation_src * representation_src, dim=1).unsqueeze(1)
        l_tgt = torch.sum(representation_tgt * representation_tgt, dim=1).unsqueeze(1)

        # cosine = inner_product / (l_src.repeat(l_src.size(0), 1).transpose(0,1).pow(0.5) * l_tgt.repeat(l_src.size(0), 1).pow(0.5))
        distance_square =  - 2 * inner_product + l_src.expand(l_src.size(0), -1).transpose(0,1) + l_tgt.expand(l_src.size(0), -1)

        # print(distance_square)
        # print(torch.diag(distance_square, 0))
        # exit()

        # cos_sort, cos_idx = torch.sort(cosine, dim=1, descending=True)
        # dis_sort, dis_idx = torch.sort(distance_square, dim=1, descending=False)
        # dis_idx_np = dis_idx.data.cpu().numpy()
        # maps = []
        # for i in range(len(dis_idx_np)):
        #     rank = np.where(dis_idx_np[i]==i)[0][0]
        #     # print(rank)
        #     if rank >= 65:
        #         maps.append(0)
        #     else:
        #         maps.append(1/(rank+1))

        # print(sum(maps)/len(maps))
        # tot_maps.append(sum(maps)/len(maps))
        tot_maps.append(torch.mean(torch.diag(distance_square, 0)).cpu().data[0])
        # print(torch.mean(torch.diag(distance_square, 0)).cpu().data[0])
        begin += batch_size
        end += batch_size
        print(end)
    # model.train()
    print(sum(tot_maps)/len(tot_maps))
    pass

def main(load_model_file_name, config_file):
    use_cuda = True
    gpu_idx = 0

    data = read_parameters(config_file)  # if c == None, use the default parameters.
    parameters = data['parameters']
    model_name = data['model_name']
    fn = data['fn']
    fn_unparallel = data['fn_unparallel']
    print("Model name:", model_name)
    print("Model file name:", load_model_file_name)

    # Pre-processing:
    vocab, sentences_ids, sentences_lengths, sentences_ids_up, sentences_lengths_up \
        = preprocessing(fn, parameters, fn_unparallel, use_cuda)

    with torch.cuda.device(gpu_idx):
        model = init_model(model_name, parameters)
        model.load_state_dict(torch.load(load_model_file_name))

        knn_eval(model, 1, 64, sentences_ids['test'][0],
                 sentences_ids['test'][1], sentences_lengths['test'][0],
                 sentences_lengths['test'][1], True)

if __name__ == "__main__":

    exit()


    models = ['123_1stHidden_gvae_z256_s2s',
              '123_1stHidden_vnmt_z256_s2s',
              '']
    model_name = ""
    load_model_file_name = model_name + ".md"
    config = os.path.join("loggings", model_name, ".config")




