import leveldb
import globals

class DBManager(object):
    description_db = None
    name_db = None
    alias_db = None
    notable_type_db = None
    type_db = None
    entity_surface_db = None
    @staticmethod
    def get_name(mid):
        if not DBManager.name_db:
            DBManager.name_db = leveldb.LevelDB(globals.config.get('LevelDB', 'name_db'))

        try:
            name = DBManager.name_db.Get(mid)
            return name.decode('utf8'), "success"
        except KeyError:
            return None, "key error"

    @staticmethod
    def get_alias(mid):
        if not DBManager.alias_db:
            DBManager.alias_db = leveldb.LevelDB(globals.config.get('LevelDB', 'alias_db'))

        try:
            alias = DBManager.alias_db.Get(mid)
            return alias.decode('utf8').split('\t'), "success"
        except KeyError:
            return None, "key error"

    @staticmethod
    def get_candidate_entities(surface, threshold):
        if not DBManager.entity_surface_db:
            DBManager.entity_surface_db = leveldb.LevelDB(globals.config.get('LevelDB', 'entity_surface_db'))

        try:

            res = DBManager.entity_surface_db.Get(surface)
            rank = []
            for e in res.split('\t'):
                mid, score = e.split()
                score = float(score)
                if score > threshold:
                    rank.append([mid, score])
            rank = sorted(rank, key=lambda x: x[1], reverse=True)
            return rank
        except KeyError:
            return []

if __name__ == '__main__':

    print DBManager.get_candidate_entities("jake'sstory", 0)

