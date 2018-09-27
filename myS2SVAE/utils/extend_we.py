import numpy as np

def read_word_embedding(we_path):
    f = open(we_path, "r")
    we = {}
    for line in f:
        line.strip().split()
        word = line[0]
        we[word] = line[1:]
    return we

def clustering_by_word_embedding(word_embedding, emb_dim, sentences, method = "ave"):
    sentences_rep = []
    for sent in sentences:
        sent_rep = np.zeros(emb_dim)
        count = 0
        for word in sent:
            if word in word_embedding:
                sent_rep += np.array(word_embedding[word])
                count += 1
        sent_rep /= count
        sentences_rep.append(sent_rep)
    from GMM import gmm
    gmm_model = gmm(200)
    gmm_model.train(sentences_rep, "cluster_emb.txt", sentences)

if __name__ == "__main__":
    import Utils
    fn_quora = {
        'trn_src': '../data/quora_duplicate_questions_trn.src',
        'trn_tgt': '../data/quora_duplicate_questions_trn.tgt',
        'dev_src': '../data/quora_duplicate_questions_dev.src',
        'dev_tgt': '../data/quora_duplicate_questions_dev.tgt',
        'test_src': '../data/quora_duplicate_questions_test.src',  # '../data/quora_duplicate_questions_test.src'
        'test_tgt': '../data/quora_duplicate_questions_test.tgt'  # '../data/quora_duplicate_questions_test.tgt'
    }
    word_embedding = read_word_embedding("../extend/glove.6B.200d.txt")
    print(len(word_embedding))
    sentences_word = Utils.read_data(fn_quora, min_str_len=1, max_str_len=128)
    clustering_by_word_embedding(word_embedding, 200, sentences_word)
