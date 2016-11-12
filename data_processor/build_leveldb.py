import sys
sys.path.insert(0, '..')
import json
import globals
from db_manager import DBManager
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
                db.Put(alias.encode('utf8'), ('%s %s' % (mid, 1.)).encode('utf8'))

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

def gen_unsolved_sentence(fn_in, fn_out):
    avg_candidate = 0
    num = 0
    with open(fn_in) as fin, open(fn_out, 'w') as fout:
        for line in fin:
            data = json.loads(line, encoding='utf8')
            gold_entity = data['entity']
            surfaces = data['predict'].split("\t")
            found = False
            candidates = []
            for surface in surfaces:
                surface = surface.lower().replace(' ', '')
                res = DBManager.get_candidate_entities(surface, 0.1)

                for e in res:
                    entity, score = e
                    candidates.append(e)
                    if entity == gold_entity:
                        avg_candidate += len(res)
                        num += 1
                        found = True
                        break
            if not found:
                # print >> fout, line.strip(), candidates
                print surfaces, data['gold']
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
