import sys
sys.path.insert(0, '..')
import json
import globals
from utils.db_manager import DBManager

import leveldb
import unicodedata

def save_description_to_leveldb(fn_description):
    db = leveldb.LevelDB('../db/description.db')
    with open(fn_description) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            db.Put(ll[0], ('\t'.join(ll[1:])).encode('utf8'))

def save_name_to_leveldb(fn_name):

    db = leveldb.LevelDB('../db/name.db')
    with open(fn_name) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 3:
                continue
            if not ll[2].endswith('@en'):
                continue
            db.Put(ll[0], ll[2][1:-4].encode('utf8'))

def save_notable_type_to_leveldb(fn_type):

    db = leveldb.LevelDB('../db/notable_type.db')
    with open(fn_type) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if not ll[0].startswith('m.'):
                continue
            if not ll[1] == 'common.topic.notable_types':
                continue
            try:
                t = db.Get(ll[0])
                t = set(t.decode('utf8').split('\t'))
                t.add(ll[2])
                db.Put(ll[0], ('\t'.join(t)).encode('utf8'))
            except KeyError:
                db.Put(ll[0], ll[2].encode('utf8'))

def save_type_to_leveldb(fn_type):

    db = leveldb.LevelDB('../db/type.db')
    with open(fn_type) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if not ll[0].startswith('m.'):
                continue
            if not ll[1] == 'type.object.type':
                continue
            try:
                t = db.Get(ll[0])
                t = set(t.decode('utf8').split('\t'))
                t.add(ll[2])
                db.Put(ll[0], ('\t'.join(t)).encode('utf8'))
            except KeyError:
                db.Put(ll[0], ll[2].encode('utf8'))

def save_alias_to_leveldb(fn_alias):

    db = leveldb.LevelDB('../db/alias.db')
    with open(fn_alias) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 2:
                continue
            mid, alias = ll
            if not mid.startswith('m.'):
                continue

            try:
                t = db.Get(mid)
                t = set(t.decode('utf8').split('\t'))
                t.add(alias)
                db.Put(mid, ('\t'.join(t)).encode('utf8'))
            except KeyError:
                db.Put(mid, alias.encode('utf8'))

def save_dict_to_leveldb(fn_alias, fn_aqqu):

    db = leveldb.LevelDB('../db/entity.surface.db')
    with open(fn_alias) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 2:
                continue
            mid, alias = ll
            if not mid.startswith('m.'):
                continue
            alias = alias.lower().replace(' ', '')
            try:
                t = db.Get(alias.encode('utf8'))
                t = set(t.decode('utf8').split('\t'))
                t.add("%s %s" % (mid, 1.))
                db.Put(alias.encode('utf8'), ('\t'.join(t)).encode('utf8'))
            except KeyError:
                db.Put(alias.encode('utf8'), ('%s %s' % (mid, 1.1)).encode('utf8'))

    with open(fn_aqqu) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 3:
                continue
            surface, score, mid = ll
            try:
                t = db.Get(surface.encode('utf8'))
                t = set(t.decode('utf8').split('\t'))
                t.add("%s %s" % (mid, score))
                db.Put(surface.encode('utf8'), ('\t'.join(t)).encode('utf8'))
            except KeyError:
                db.Put(surface.encode('utf8'), ('%s %s' % (mid, score)).encode('utf8'))

def save_mediator_relation_to_leveldb(fn):

    db = leveldb.LevelDB('../db/mediator-relations')
    with open(fn) as fin:
        for line in fin:
            rel = line.decode('utf8').strip()
            if  rel.startswith('m.'):
                continue
            try:
                r = db.Get(rel.encode('utf8'))
            except:
                db.Put(rel.encode('utf8'))


def add_key_to_dict_leveldb(fn_alias, key_fn_list):
    db = leveldb.LevelDB('../db/entity.surface.db')
    with open(fn_alias) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 2:
                continue
            mid, alias = ll
            if not mid.startswith('m.'):
                continue
            for key_fn in key_fn_list:
                alias, score = key_fn(alias)
                try:
                    t = db.Get(alias.encode('utf8'))
                    t = set(t.decode('utf8').split('\t'))
                    t.add("%s %s" % (mid, score))
                    db.Put(alias.encode('utf8'), ('\t'.join(t)).encode('utf8'))
                except KeyError:
                    db.Put(alias.encode('utf8'), ('%s %s' % (mid, score)).encode('utf8'))

from corenlp_parser.parser import CoreNLPParser
def get_lemma_fn():
    parser = CoreNLPParser.init_from_config()
    def fn(text):
        text = text.lower()
        tokens = parser.parse(text)

        lemma = [t.lemma if t.pos.startswith('NN') else t.token for t in tokens]
        return ''.join(lemma), 0.9
    return fn

def other_to_english(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').lower().replace(' ', ''), 0.9
import re
valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
                                           r'RB|CC|NNP.?|NN.?|JJ.?|CD|DT|MD|'
                                           r'POS)+$')
ignore = {'are', 'is', 'were', 'was', 'be', 'of', 'the', 'and', 'or', 'a', 'an'}


def is_entity_occurrence(all_pos, all_token, start, end):

    # Concatenate POS-tags
    token_list = all_token[start:end]
    pos_list = all_pos[start:end]
    pos_str = ''.join(pos_list)
    # Check if all tokens are in the ignore list.
    # For length 1 only allows nouns
    if all((t in ignore for t in token_list)):
        return False

    if len(pos_list) == 1 and (pos_list[0].startswith('N') or pos_list[0].startswith('JJ')) \
            or len(pos_list) > 1 and valid_entity_tag.match(pos_str):
        if len(pos_list) == 1:
            if pos_list[0].startswith('NNP') and start > 0 and all_pos[start - 1].startswith('NNP'):
                return False
            elif pos_list[-1].startswith('NNP') and end < len(all_pos) and all_pos[end].startswith('NNP'):
                return False
        return True
    return False



def find_word(sentence, word):
    word_len = len(word)
    sentence_len = len(sentence)
    text = ' '.join(word)
    for i in range(sentence_len - word_len+1):
        if sentence[i] == word[0] and ' '.join(sentence[i:i+word_len]) == text:
            return i
    return -1

def gen_unsolved_sentence(fn_in, fn_out):
    avg_candidate = 0
    num = 0
    with open(fn_in) as fin, open(fn_out, 'w') as fout:
        for line in fin:
            data = json.loads(line, encoding='utf8')
            gold_entity = data['entity']
            surfaces = data['predict'].split("\t")

            candidates = dict()
            for surface in surfaces:
                surface = surface.lower().replace(' ', '')
                res = DBManager.get_candidate_entities(surface, 0.1)

                for e in res:
                    if e[0] not in candidates or e[1] > candidates[e[0]]:
                        candidates[e[0]] = e[1]
            if len(candidates) == 0:
                sentence =[w.split('|')[0]for w in data['tag_res'].split()][1:-1]
                all_pos = data['pos'][1:-1]
                #use ngram of surface

                for surface in surfaces:
                    surface = surface.lower().split()
                    if len(surface) == 0:
                        continue
                    start = find_word(sentence, surface)
                    if start == -1:
                        continue
                    l = len(surface)
                    found = False
                    for j in range(l, 0, -1):
                        # if found:
                        #     break
                        for i in range(l-j+1):
                            if is_entity_occurrence(all_pos, sentence, start + i, start + i + j):
                                s = ''.join(surface[i:i+j])
                                res = DBManager.get_candidate_entities(s, 0.1)
                                for e in res:
                                    if e[1] < 1.1 and (e[0] not in candidates or e[1] > candidates[e[0]]):
                                        candidates[e[0]] = e[1]
                                found = len(res) > 0
            # candidates = sorted(candidates.items(), key=lambda x:x[1], reverse=True)[:20]
            candidates = candidates.items()
            correct = False
            for e, _ in candidates:
                if e == gold_entity:
                    avg_candidate += len(candidates)
                    num += 1
                    correct = True
                    break
            if not correct:
                # print >> fout, line.strip(), candidates
                print surfaces, data['gold'].split('\t'), gold_entity
            # else:
                # print line.strip()
                # print candidates
    print "%s find correct topic entity" % num
    print "average number of candidate entities: %s" % (avg_candidate * 1.0 / num)


if __name__ == '__main__':
    # save_description_to_leveldb('../data/description.sentences.large.clean')
    # save_name_to_leveldb('../data/nameInFb')
    # save_notable_type_to_leveldb('../data/typeInFb')
    # save_type_to_leveldb('../data/typeInFb')

    # save_alias_to_leveldb('../../data/fb.alias')
    # save_dict_to_leveldb('../../data/fb.alias', '../../aqqu/data/entity-surface-map')
    globals.read_configuration('../config.cfg')
    gen_unsolved_sentence(sys.argv[1], sys.argv[2])

    # add_key_to_dict_leveldb('../../data/fb.alias', [get_lemma_fn(), other_to_english])
