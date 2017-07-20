
import pickle


class VocabMapping(object):
    def __init__(self):
        with open("data/vocab.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())

    def get_index(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]

    def get_size(self):
        return len(self.dic)
