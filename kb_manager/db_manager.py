import sys
sys.path.insert(0, '..')
import leveldb
import globals

class DBManager(object):
    description_db = None
    name_db = None
    alias_db = None
    notable_type_db = None
    type_db = None
    entity_surface_db = None
    freebase_db = None
    mediate_relations = None

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
            if not isinstance(surface, str):
                surface = surface.encode('utf8')
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

    @staticmethod
    def get_one_hop_path(subject):
        if not DBManager.freebase_db:
                DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))
        try:
            values = DBManager.freebase_db.Get(subject.encode('utf8')).decode('utf8')

            values = values.split('\t')
            ret = []
            for value in values:
                value = value.split()
                for v in value[1:]:
                    ret.append([subject, value[0], v])
            return ret

        except KeyError:
            return []

    # @staticmethod
    # def get_one_hop_relation(subject):
    #     if not DBManager.freebase_db:
    #         DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))
    #     try:
    #         multi_rel_objs = DBManager.freebase_db.Get(subject.encode('utf8')).decode('utf8')
    #
    #         multi_rel_objs = multi_rel_objs.split('\t')
    #         ret = []
    #         for rel_objs in multi_rel_objs:
    #             ret.append(rel_objs.split()[0])
    #         return ret
    #     except KeyError:
    #         return []

    @staticmethod
    def is_mediate_relation(relation):
        if not DBManager.mediate_relations:
            DBManager.mediate_relations = set()
            with open(globals.config.get('FREEBASE', 'mediator-relations')) as fin:
                for line in fin:
                    rel = line.decode('utf8').strip()
                    if not rel.startswith('m.'):
                        DBManager.mediate_relations.add(rel)

        return relation in DBManager.mediate_relations

    @staticmethod
    def get_subgraph(subject):
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))

        first_hop = DBManager.get_one_hop_path(subject)
        ret = []
        for i in xrange(len(first_hop)):
            r1 = first_hop[i][1]
            if DBManager.is_mediate_relation(r1):
                mediate_node = first_hop[i][2]
                second_hop = DBManager.get_one_hop_path(mediate_node)
                # # if CVT node has too much neighbors, ignore it (Maybe it isn't CVT)
                # if len(second_hop) > 10:
                #     continue
                for j in xrange(len(second_hop)):
                    if second_hop[j][2] == subject:
                        continue
                    ret.append([first_hop[i], second_hop[j]])
            else:
                ret.append([first_hop[i]])
        return ret

    @staticmethod
    def get_multiple_hop_relations(subject):
        """
        return list of relations from subject. for example,  [[r], [r_1, r_2], ..., ].
        :param subject:
        :return:
        """
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))
        subgraph = DBManager.get_subgraph(subject)
        vis = set()
        ret = []
        for path in subgraph:
            if len(path) == 1 and path[0][1] not in vis:
                vis.add(path[0][1])
                ret.append([path[0][1]])
            elif len(path) == 2 and (path[0][1], path[1][1]) not in vis:
                vis.add((path[0][1], path[1][1]))
                ret.append([path[0][1], path[1][1]])
        return ret

    @staticmethod
    def get_core_paths_without_object(subject):
        """
        TODO: check whether answer has name
        :param subject:
        :return: 2D list
        """
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))

        first_hop = DBManager.get_one_hop_path(subject)
        ret = []
        vis = set()
        for i in xrange(len(first_hop)):
            r1 = first_hop[i][1]
            if DBManager.is_mediate_relation(r1):
                mediator = first_hop[i][2]
                second_hop = DBManager.get_one_hop_path(mediator)
                for j in xrange(len(second_hop)):
                    if second_hop[j][2] == subject:
                        continue
                    if (subject, r1, mediator, second_hop[j][1]) not in vis:
                        vis.add((subject, r1, mediator, second_hop[j][1]))
                        ret.append([subject, r1, mediator, second_hop[j][1]])
            else:
                if (subject, first_hop[i][1]) not in vis:
                    vis.add((subject, first_hop[i][1]))
                    ret.append([subject, first_hop[i][1]])
        return ret


if __name__ == '__main__':
    globals.read_configuration('../config.cfg')
    # print DBManager.get_candidate_entities("jake'sstory", 0)
    # print len(DBManager.get_one_hop_graph('m.0f2y0'))
    print DBManager.get_subgraph('m.0f2y0')
    print DBManager.get_multiple_hop_relations('m.0f2y0')

