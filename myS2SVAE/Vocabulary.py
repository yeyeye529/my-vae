from collections import defaultdict
import logging

class vocabulary(object):
    def __init__(self, max_vocab_size = 1000000):
        self._vocab = None
        self._word2idx = None
        self.bos_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.pad_id = 3
        self._max_vocab_size = max_vocab_size
        self.logger = logging.getLogger('vocabulary')
        pass

    @property
    def vocab(self):
        if self._vocab == None:
            assert 'Vocabulary does not exist'
            return None
        else:
            return self._vocab

    @property
    def word2idx(self):
        if self._word2idx == None:
            assert 'Vocabulary does not exist'
            return None
        else:
            return self._word2idx

    def build_vocab(self, sentence_list, remove_low_freq = 0):
        vocab = []
        vocab_with_cnt = defaultdict(int)

        for s in sentence_list:
            for w in s:
                vocab_with_cnt[w] += 1

        print("Original vocab size = ", len(vocab_with_cnt))
        self.logger.info("Original vocab size = " + str(len(vocab_with_cnt)))
        i = 0
        for w, cnt in vocab_with_cnt.items():
            if cnt > remove_low_freq:
                vocab.append(w)
                i += 1
            if i > self._max_vocab_size:
                break
        print("Now vocab size = ", len(vocab))
        self.logger.info("Now vocab size = " + str(len(vocab)))
        vocab.sort()
        vocab_new = ['<s>', '</s>', '<unk>', '<pad>']  # <s>: begin of sentence; </s>: end of sentence; <unk>: unknown word
        vocab_new += vocab
        word2idx = {w:i for i, w in enumerate(vocab_new)}
        self._vocab = vocab_new
        self._word2idx = word2idx


    def vocab_sent(self, sent_list, max_str_len = None, truncated = False, truncated_length = 0):
        sent_list_new = list()
        sent_lengths = []
        if max_str_len == None:
            max_str_len = 0
            for i in sent_list:
                if len(i) > max_str_len:
                    max_str_len = len(i)
            max_str_len += 2  # + <bos>, <eos>
        if truncated:
            max_str_len = truncated_length + 2  # + <bos>, <eos>
        for s in sent_list:
            sent_new = list()
            sent_new.append(self.bos_id)
            for w in s:
                if w in self._word2idx:
                    sent_new.append(self._word2idx[w])
                else:
                    sent_new.append(self.unk_id) # <unk>
            sent_new.append(self.eos_id)
            sent_lengths.append(min(len(sent_new), max_str_len))
            #Padding:
            if (len(sent_new) < max_str_len):
                sent_new.extend([self.pad_id]*(max_str_len - len(sent_new)))
            else:
                sent_new = sent_new[0:max_str_len]

            sent_list_new.append(sent_new)

        return sent_list_new, sent_lengths, max_str_len


    def vocab_sent_for_multi_refs(self, sent_list, max_str_len = None, truncated = False, truncated_length = 0):
        sent_list_new = list()
        sent_lengths = []
        if max_str_len == None:
            max_str_len = 0
            for refs in sent_list:
                for i in refs:
                    if len(i) > max_str_len:
                        max_str_len = len(i)
            max_str_len += 2  # + <bos>, <eos>
        if truncated:
            max_str_len = truncated_length + 2  # + <bos>, <eos>

        for refs in sent_list:
            refs_new = list()
            refs_lengths = list()
            for s in refs:
                sent_new = list()
                sent_new.append(self.bos_id)
                for w in s:
                    if w in self._word2idx:
                        sent_new.append(self._word2idx[w])
                    else:
                        sent_new.append(self.unk_id) # <unk>
                sent_new.append(self.eos_id)
                refs_lengths.append(min(len(sent_new), max_str_len)) # sent_lengths
                #Padding:
                if (len(sent_new) < max_str_len):
                    sent_new.extend([self.pad_id]*(max_str_len - len(sent_new)))
                else:
                    sent_new = sent_new[0:max_str_len]
                refs_new.append(sent_new)   # sent_list_new
            sent_list_new.append(refs_new)
            sent_lengths.append(refs_lengths)


        return sent_list_new, sent_lengths, max_str_len
