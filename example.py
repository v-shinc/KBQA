

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
            neg_entities -= set(data['mid1'])
            instance = {'q': data['utterance'], 'pos': data['mid1'], 'neg': data['mids'].values()}
            print >> fout, json.dumps(instance, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
    gen_description_data(['../data/wq.dev.complete.v2', '../data/wq.train.complete.v2'], '../data/entity.train')