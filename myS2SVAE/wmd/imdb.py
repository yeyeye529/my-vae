

from myS2SVAE.wmd import get_distance
from gensim import utils
import random



class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        with utils.smart_open(self.sources[0]) as f1, utils.smart_open(self.sources[1]) as f2, utils.smart_open(self.sources[2]) as f3:
            for item_no, line, bleu in enumerate(zip(f1,f2,f3)):
                yield LabeledSentence(line[0].split(),line[1].split(),[item_no,bleu])

    def to_array(self):
        self.sentences = []
        self.bleus = []
        with utils.smart_open(self.sources[0]) as f1, utils.smart_open(self.sources[1]) as f2, utils.smart_open(self.sources[2]) as f3:
            for item_no, line in enumerate(zip(f1, f2,f3)):
                self.sentences += [line[0:2]]
                self.bleus.append(float(line[2]))
        return self.sentences, self.bleus

length = 2000

# 'data/train-neg.txt':'TRAIN_NEG', 
sources = ['../data/quora_duplicate_questions_test.tgt', '../attention_weighted_test.txt', "../attention_weighted_test.bleu"]


LS = LabeledLineSentence(sources)

sentences, bleus = LS.to_array()

#sentences = random.sample(LabeledLineSentence(sources).to_array(), 14950)

 #for i in sentences:
#     print(i)
target = [line[0].strip() for line in sentences]
output = [line[1].strip() for line in sentences]

# for i in zip(output, target):
#      print(i)







#sample1 = "this show was incredible i ve seen all three and this is the best this movie has suspense a bit of romance stunts that will blow your mind go bobbie great characters and amazing locations where was this filmed will there be more i really liked the story line with her brother looking forward to chameleon and to see how the world is saved yet again"
#sample2 = "this anime was underrated and still is hardly the dorky kids movie as noted i still come back to this years after i first saw it one of the better movies released the animation while not perfect is good camera tricks give it a d feel and the story is still as good today even after i grew up and saw ground breakers like neon genesis evangelion and rahxephon it has nowhere near the depth obviously but try to see it from a lighthearted view it s a story to entertain not to question still one of my favourites i come back too when i feel like a giggle on over more lighthearted animes not to say its a childish movies there are surprisingly sad moments in this and you need a sense of humour to see it all"

#target = sample1[:length]
#print("target:",len(target.split()))

scores = {}
#target = ["why should we not hire you"]
#output = ["why should i not hire you"]
for index, sentence in enumerate(zip(target, output)):
	
    scores[index] = get_distance(sentence[0], sentence[1])
    print("target:",sentence[0].decode('utf-8'))
    print("output:",sentence[1].decode('utf-8'))
    print(scores[index],"\t",bleus[index])
    print("\n\n")

# for k,v in scores.items():
#     print(k,v)

#sorted_list = sorted(scores.items(), key=lambda kv: kv[1])
#top = sorted_list[:20]

#print(target)
# for index, score in top:
#     print("\n")
#     print(sentences[index])
