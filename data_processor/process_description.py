
import json
import os

import jsonrpclib
from string import punctuation
from pprint import pprint

from corenlp import StanfordCoreNLP

def split_one(params):
    index, fn, start, end = params
    # skip 'start' line
    lno = 0
    fin = open(fn)
    while lno < start:
        fin.readline()
        lno += 1

    ret = []
    parser = StanfordCoreNLP("stanford-corenlp-full-2015-01-29/", properties="default.properties", serving=False)
    for i in xrange(start, end):
        line = fin.readline()

        ll = line.decode('utf8').strip().split('\t')
        if len(ll) != 3:
            continue
        if not ll[2].endswith('@en'):
            continue
        text = ll[2][1:-4]

        text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\', ' ')
        try:
            rsp = json.loads(parser.parse(text))

            sentences = []
            for s in rsp['sentences']:
                sentences.append(s['text'])
            ret.append(('%s\t%s' % (ll[0], '\t'.join(sentences))).encode('utf8'))
        except Exception as e:
            print e

    fin.close()
    return ret


def split_main(fn_in, fn_out):
    from multiprocessing import Pool
    MAX_POOL_NUM = 1

    num_line = 0
    with open(fn_in) as fin:
        for _ in fin:
            num_line += 1
    print "There are %d lines to process." % num_line
    chunk_size = 500
    parameters = []
    i = 0
    while i * chunk_size < num_line:
        parameters.append((i, fn_in, i * chunk_size, min(num_line, (i + 1) * chunk_size)))
        i += 1

    pool = Pool(MAX_POOL_NUM)
    ret_list = pool.imap_unordered(split_one, parameters)
    pool.close()
    with open(fn_out, 'w') as fout:
        for l in ret_list:
            for s in l:
                print >> fout, s
    pool.join()

def local_split_description(fn_in, fn_out):
    parser = StanfordCoreNLP("stanford-corenlp-full-2015-01-29/", properties="default.properties", serving=False)
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) != 3:
                    continue
                if not ll[2].endswith('@en'):
                    continue
                text = ll[2][1:-4]

                text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\', ' ')
                try:
                    rsp = json.loads(parser.parse(text))

                    sentences = []
                    for s in rsp['sentences']:
                        sentences.append(s['text'])

                    print >> fout, ('%s\t%s' % (ll[0], '\t'.join(sentences))).encode('utf8')
                except Exception as e:
                    print e.message



class StanfordNLP:
    def __init__(self, port_number=8080):
        self.server = jsonrpclib.Server("http://localhost:%d" % port_number)

    def parse(self, text):
        return json.loads(self.server.parse(text))

def split_into_sentences(parser, article):
    rsp = parser.parse(article)
    sentences = []
    for s in rsp['sentences']:
        sentences.append(s['text'])
    return sentences

def split_description(fn_in, fn_out):
    parser = StanfordNLP()
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) != 3:
                    continue
                if not ll[2].endswith('@en'):
                    continue
                text = ll[2][1:-4]

                text = text.replace('\\\n', ' ').replace('\\', '')

                sentences = split_into_sentences(parser, text)
                print >> fout, ('%s\t%s' % (ll[0], '\t'.join(sentences))).encode('utf8')

def clean_descrption(fn_in, fn_out):
    punct = set(punctuation)
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                cleaned = []
                for s in ll[1:]:
                    cleaned.append(''.join([c for c in s if c not in punct]))
                print >> fout, ('%s\t%s' % (ll[0], '\t'.join(cleaned).lower())).encode('utf8')


if __name__ == '__main__':
    # text = 'Holiday is a 2014 Action Romance Thriller film written by A.R. Murugadoss and directed by A.R. Murugadoss. Soldier is a 2006 Short Animation film directed and written by  David Peros Bonnot and Simon Bogojevic-Narath.'
    # nlp = StanfordNLP()
    # result = nlp.parse(text)
    # result = split_into_sentences(nlp, text)
    # split_description('../data/descriptionInTriple', '../data/description.sentences')
    # clean_descrption('../data/description.sentences', '../data/description.sentences.clean')
    # local_split_description('descriptionInTriple', 'description.sentences.tmp')
    split_main('descriptionInTriple', 'description.sentences.mt')