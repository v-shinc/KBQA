from nltk.parse.stanford import StanfordParser


class NLPParser(object):
    def __init__(self):
        self.eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

    def tag_pos(self, text):
        res = list(self.eng_parser.parse(text.split(' ')))[0].pos()
        return [t[0] for t in res], [t[1] for t in res]