import os


import torch
from torch.autograd import Variable
import numpy as np
from myS2SVAE import chinese_exp


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
        encoder_input = Variable(torch.LongTensor(src_list[begin: end]))
        batch_src_lengths = src_lengths[begin: end]
        # print(batch_src_lengths)
        encoder_input = encoder_input.cuda() if use_cuda else encoder_input

        decoder_input = Variable(torch.LongTensor(tgt_list[begin: end])).squeeze()
        batch_tgt_lengths = tgt_lengths[begin: end]
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        batch_tgt_lengths_np = np.array(batch_tgt_lengths).transpose()[0]

        encoding_outputs = model.encoding(encoder_input, decoder_input, batch_src_lengths, batch_tgt_lengths_np,
                                          cal_tgt = True)
        representation_src = encoding_outputs['representation_src'].squeeze()
        representation_tgt = encoding_outputs['representation_tgt'].squeeze()
        # print(representation_src.size())
        # print(representation_tgt.size())
        inner_product = torch.matmul(representation_src, representation_tgt.transpose(0,1))

        l_src = torch.sum(representation_src * representation_src, dim=1)
        l_tgt = torch.sum(representation_tgt * representation_tgt, dim=1)

        # cosine = inner_product / (l_src.pow(0.5) * l_tgt.pow(0.5))
        distance_square =  - 2 * inner_product + l_src.repeat(l_src.size(0), 1).transpose(0,1) + l_tgt.repeat(l_src.size(0), 1)

        # cos_sort, cos_idx = torch.sort(cosine, dim=1, descending=True)
        dis_sort, dis_idx = torch.sort(distance_square, dim=1, descending=False)


        dis_idx_np = dis_idx.data.cpu().numpy()
        maps = []
        for i in range(len(dis_idx_np)):
            rank = np.where(dis_idx_np[i]==i)[0][0]
            if rank >= 10:
                maps.append(0)
            else:
                maps.append(1/(rank+1))

        # print(sum(maps)/len(maps))
        tot_maps.append(sum(maps)/len(maps))
        begin += batch_size
        end += batch_size
    model.train()
    print(sum(tot_maps)/len(tot_maps))
    pass

def main(load_model_file_name, config_file):
    use_cuda = True
    gpu_idx = 0

    data = chinese_exp.read_parameters(config_file)  # if c == None, use the default parameters.
    parameters = data['parameters']
    model_name = data['model_name']
    fn = data['fn']
    fn_unparallel = data['fn_unparallel']
    print("Model name:", model_name)
    print("Model file name:", load_model_file_name)

    # Pre-processing:
    vocab, sentences_ids, sentences_lengths, sentences_ids_up, sentences_lengths_up \
        = chinese_exp.preprocessing(fn, parameters, fn_unparallel, use_cuda)

    with torch.cuda.device(gpu_idx):
        model = chinese_exp.init_model(model_name, parameters)
        model.load_state_dict(torch.load(load_model_file_name))
        knn_eval(model, 1, 64, sentences_ids['test'][0],
                 sentences_ids['test'][1], sentences_lengths['test'][0],
                 sentences_lengths['test'][1], True)

if __name__ == "__main__":

    models = [#'123_1stHidden_gvae_z256_s2s2',
              #'123_1stHidden_vnmt_z256_s2s2',
              '123_1stHidden_debugdebug_k100_z256_re_s2s_noKA_noMax2',
              '123_1stHidden_debugdebug_k1_z256_re_s2s_noKA1',
              '123_1stHidden_debugdebug_k1_z256_re_s2s_noKA_fix1',]
    for model_name in models:
        print("+++++")
        print(model_name)
        load_model_file_name = model_name + ".md"
        config = os.path.join("loggings", model_name + ".config")
        print("+++++")
        main(load_model_file_name, config)




