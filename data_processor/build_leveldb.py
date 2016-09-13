def save_description_to_leveldb(fn_description):
    import leveldb
    db = leveldb.LevelDB('../db/description.db')
    with open(fn_description) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            db.Put(ll[0], ('\t'.join(ll[1:])).encode('utf8'))

def save_name_to_leveldb(fn_name):
    import leveldb
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
    import leveldb
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
    import leveldb
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

if __name__ == '__main__':
    # save_description_to_leveldb('../data/description.sentences.large.clean')
    # save_name_to_leveldb('../data/nameInFb')
    # save_notable_type_to_leveldb('../data/typeInFb')
    save_type_to_leveldb('../data/typeInFb')