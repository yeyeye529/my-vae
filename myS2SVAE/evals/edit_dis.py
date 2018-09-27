import numpy

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = numpy.array(tuple(source))
    target = numpy.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = numpy.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = numpy.minimum(
            current_row[1:],
            numpy.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = numpy.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def cal_levenshtein(refs, hyps):
    if len(refs) != len(hyps):
        print("Error! Length does not match.")
    s_list = []
    for sid in range(len(refs)):
        s = levenshtein(hyps[sid], refs[sid])
        s_list.append(s)

    return (sum(s_list)/len(s_list))

def read_data(file_list):
    res = []
    for fn in file_list:
        sents = []
        f = open(fn, "r", encoding="utf-8")
        for line in f:
            s = line.split()
            if s[-1] == "</s>":
                sents.append(s[2:len(s)-1])
            else:
                sents.append(s[2:])
        res.append(sents)
        f.close()
    return res

def cal_levenshtein_toEachOther(file_list):
    for fid, fn in enumerate(file_list):
        f = open(fn, "r", encoding="utf-8")
        file_res = []
        sents = []
        for i, line in enumerate(f):
            if i % 5 == 0:
                sents = []
            else:
                s = line.strip().split()
                if s[-1] == "</s>":
                    sents.append(s[2:len(s)-1])
                else:
                    sents.append(s[2:])
            if i % 5 == 4:
                res = []
                for k in range(len(sents)):
                    for j in range(len(sents)):
                        if k == j:
                            continue
                        res.append(levenshtein(sents[k], sents[j]))
                file_res.append(sum(res) / len(res))
                # print(sum(res) / len(res))
        print(fid, sum(file_res) / len(file_res))

def cal_levenshtein_toOri(file_list, ref_sents, m="ave"):
    for fid, fn in enumerate(file_list):
        f = open(fn, "r", encoding="utf-8")
        file_res = []
        sents = []
        for i, line in enumerate(f):
            if i % 5 == 0:
                sents = []
            else:
                s = line.strip().split()
                if s[-1] == "</s>":
                    sents.append(s[2:len(s)-1])
                else:
                    sents.append(s[2:])
            if i % 5 == 4:
                res = []
                for k in range(len(sents)):
                    res.append(levenshtein(sents[k], ref_sents[i//5]))
                file_res.append(sum(res) / len(res))
                # print(sum(res) / len(res))
        print(fid, sum(file_res) / len(file_res))


if __name__ == '__main__':
    ref_file = ["../../data/quora_duplicate_questions_test.src.2"]
    hyp_files = ["../hyp_123_1stHidden_gvae_z256_s2s.txt",
                 "../hyp_123_1stHidden_vnmt_z256_s2s.txt",
                 # "../hyp_123_1stHidden_debugdebug_k500_z256_re_s2s_noKA.txt",
                 "../hyp_123_1stHidden_debugdebug_k1_z256_re_s2s_noKA.txt",
                 "../hyp_123_1stHidden_debugdebug_k1_z256_re_s2s_noKA_fix.txt",
                 "../hyp_123_1stHidden_debugdebug_k100_z256_re_s2s_noKA_noMax.txt",
                 "../../mySeq2Seq/hyp_seq2seq_tl=50.txt",
                 "../../data/quora_duplicate_questions_test.tgt.2"]
    all_trans_files = ["../all_trans_123_1stHidden_gvae_z256_s2s.txt",
                 "../all_trans_123_1stHidden_vnmt_z256_s2s.txt",
                 "../all_trans_123_1stHidden_debugdebug_k100_z256_re_s2s_noKA_noMax.txt",
                 "../all_trans_123_1stHidden_debugdebug_k1_z256_re_s2s_noKA.txt",
                 "../all_trans_123_1stHidden_debugdebug_k1_z256_re_s2s_noKA_fix.txt",]
                 # "../all_trans_123_1stHidden_debugdebug_k500_z256_re_s2s_noKA_noMAX.txt"]

    cal_levenshtein_toEachOther(all_trans_files)


    ref_sents = read_data(ref_file)[0]
    hyp_sents = read_data(hyp_files)
    # cal_levenshtein_toOri(all_trans_files, ref_sents)
    for i, h in enumerate(hyp_files):
        print("hyp:" + str(i) + " " + str(cal_levenshtein(ref_sents, hyp_sents[i])))
    exit()

