

from corenlp_parser.parser import CoreNLPParser
from entity_linker.entity_linker import EntityLinker
import globals

import json


def gen_description_data(fn_wq_list, fn_out):

    globals.read_configuration("config.cfg")
    entity_linker = EntityLinker.init_from_config()
    parser = CoreNLPParser.init_from_config()

    fout = open(fn_out, 'w')
    for fn_wq in fn_wq_list:
        wq = json.load(open(fn_wq), encoding='utf8')
        for data in wq:
            tokens = parser.parse(data['utterance'])
            entities = entity_linker.identify_entities_in_tokens(tokens)
            neg_entities = set()
            for e in entities:
                mid = e.get_mid()
                if mid == '':
                    continue
                if mid.startswith('m.'):
                    neg_entities.add(mid)
                else:
                    print mid, e.name, data['utterance']
            neg_entities -= set([data['mid1']])
            instance = {'q': data['utterance'], 'pos': data['mid1'], 'neg': list(neg_entities)}
            print >> fout, json.dumps(instance, ensure_ascii=False).encode('utf8')
    fout.close()

def link_entity_one(params):
    fn, start, end = params
    lno = 0
    fin = open(fn)
    while lno < start:
        fin.readline()
        lno += 1
    globals.read_configuration("config.cfg")
    parser = CoreNLPParser.init_from_config()
    ret = []

    for i in xrange(start, end):
        line = fin.readline()
        ll = line.decode('utf8').strip().split('\t')
        if len(ll) != 5:
            continue
        tokens = parser.parse(ll[4])
        ret.append((ll[4], ll[0], tokens))

    fin.close()
    return ret

def link_entity_in_simple_question_mt(fn_in, fn_out):
    from multiprocessing import Pool
    MAX_POOL_NUM = 8

    num_line = 0
    with open(fn_in) as fin:
        for _ in fin:
            num_line += 1
    print "There are %d lines to process." % num_line
    chunk_size = 50
    parameters = []
    i = 0
    while i * chunk_size < num_line:
        parameters.append((fn_in, i * chunk_size, min(num_line, (i + 1) * chunk_size)))
        i += 1

    pool = Pool(MAX_POOL_NUM)
    ret_list = pool.imap_unordered(link_entity_one, parameters)
    pool.close()
    globals.read_configuration("config.cfg")
    entity_linker = EntityLinker.init_from_config()
    with open(fn_out, 'w') as fout:
        for l in ret_list:
            for sentence, entity, tokens in l:
                entities = entity_linker.identify_entities_in_tokens(tokens)
                neg_entities = set()
                for e in entities:
                    mid = e.get_mid()
                    if mid == '':
                        continue
                    if mid.startswith('m.'):
                        neg_entities.add(mid)
                    else:
                        print mid, e.name, sentence
                neg_entities -= set([entity])
                print >> fout, json.dumps({'q': sentence, 'pos': entity, 'neg': list(neg_entities)}, ensure_ascii=False).encode('utf8')
    pool.join()

def link_entity_in_simple_question(fn_in, fn_out):
    globals.read_configuration("config.cfg")
    entity_linker = EntityLinker.init_from_config()
    parser = CoreNLPParser.init_from_config()
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) != 5:
                    continue
                tokens = parser.parse(ll[4])
                entities = entity_linker.identify_entities_in_tokens(tokens)
                neg_entities = set()
                for e in entities:
                    mid = e.get_mid()
                    if mid == '':
                        continue
                    if mid.startswith('m.'):
                        neg_entities.add(mid)
                    else:
                        print mid, e.name, ll[4]
                neg_entities -= set([ll[0]])
                line = json.dumps({'q': ll[4], 'pos': ll[0], 'neg': list(neg_entities)}, ensure_ascii=False).encode('utf8')
                print >> fout, line


if __name__ == '__main__':
    # gen_description_data(['data/wq.dev.complete.v2', 'data/wq.train.complete.v2'], 'data/wq.entity.train')
    # gen_description_data(['data/wq.test.complete.v2'], 'data/wq.entity.test')
    link_entity_in_simple_question_mt('data/simple.test.el.v2', 'data/sq.entity.test')
    # link_entity_in_simple_question_mt('data/simple.train.dev.el.v2', 'data/sq.entity.train')
