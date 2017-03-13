import sys
sys.path.insert(0, '..')
import leveldb
import globals

class DBManager(object):
    description_db = None
    name_db = None
    alias_db = None
    notable_type_db = None
    notable_for_db = None
    type_db = None
    entity_surface_db = None
    freebase_db = None
    mediator_relations = None
    mediator_nodes = None

    @staticmethod
    def get_description(mid):
        if not DBManager.description_db:
            DBManager.description_db = leveldb.LevelDB(globals.config.get('LevelDB', 'description_db'))
        try:
            return DBManager.description_db.Get(mid)
        except KeyError:
            return "No description!"

    @staticmethod
    def get_notable_type(mid):
        if not DBManager.notable_type_db:
            DBManager.notable_type_db = leveldb.LevelDB(globals.config.get('LevelDB', 'notable_type_db'))
        try:
            notable_type = DBManager.notable_type_db.Get(mid)
            return notable_type.split('\t')
        except KeyError:
            return []

    @staticmethod
    def get_notable_for(mid):
        if not DBManager.notable_for_db:
            DBManager.notable_for_db = leveldb.LevelDB(globals.config.get('LevelDB', 'notable_for_db'))
        try:
            notable_for = DBManager.notable_for_db.Get(mid)
            return notable_for.split('\t')
        except KeyError:
            return []

    @staticmethod
    def get_type(mid):
        if not DBManager.type_db:
            DBManager.type_db = leveldb.LevelDB(globals.config.get('LevelDB', 'type_db'))
        try:
            types = DBManager.type_db.Get(mid)
            return types.split('\t')
        except KeyError:
            return []

    @staticmethod
    def get_name(mid):
        if not DBManager.name_db:
            DBManager.name_db = leveldb.LevelDB(globals.config.get('LevelDB', 'name_db'))
        try:
            if not mid.startswith('m.'):
                return None, "key error"
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

    @staticmethod
    def get_one_hop_path_as_dict(subject):
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))
        try:
            values = DBManager.freebase_db.Get(subject.encode('utf8')).decode('utf8')

            values = values.split('\t')
            ret = dict()
            for value in values:  # value is "rel obj1 obj2 ..."
                value = value.split()
                rel = value[0]
                objs = value[1:]
                ret[rel] = objs
            return ret

        except KeyError:
            return None

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
    def is_mediator_relation(relation):
        if not DBManager.mediator_relations:
            DBManager.mediator_relations = set()
            with open(globals.config.get('FREEBASE', 'mediator-relations')) as fin:
                for line in fin:
                    rel = line.decode('utf8').strip()
                    if not rel.startswith('m.'):
                        DBManager.mediator_relations.add(rel)

        return relation in DBManager.mediator_relations
    @staticmethod
    def is_mediator_node(entity):
        if not DBManager.mediator_nodes:
            DBManager.mediator_nodes = set()
            with open(globals.config.get('FREEBASE', 'mediator-entities')) as fin:
                for line in fin:
                    node = line.decode('utf8').strip()
                    DBManager.mediator_nodes.add(node)
        return entity in DBManager.mediator_nodes

    @staticmethod
    def get_subgraph(subject):
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))

        first_hop = DBManager.get_one_hop_path(subject)
        ret = []
        for i in xrange(len(first_hop)):
            r1 = first_hop[i][1]
            if DBManager.is_mediator_relation(r1): #and DBManager.is_mediator_node(first_hop[i][2]):
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
    def get_subgraph_as_dict(subject):
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))
        first_hop = DBManager.get_one_hop_path_as_dict(subject)  # first_hop is a dict
        if not first_hop:
            return dict()
        new_sugraph = dict()
        for r in first_hop.keys():

            if DBManager.is_mediator_relation(r):
                for i, cvt in enumerate(first_hop[r]):
                    detail_dict = DBManager.get_one_hop_path_as_dict(cvt) # use detail dictionary to replace cvt node
                    if detail_dict:
                        if r not in new_sugraph:
                            new_sugraph[r] = list(:w)
                        new_sugraph[r].append(detail_dict)
            else:
                new_sugraph[r] = first_hop[r]
        return new_sugraph

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
            if DBManager.is_mediator_relation(r1):
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

    @staticmethod
    def get_object(path, constraints):  #TODO
        subject = path[0]

        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))

        first_hop = DBManager.get_one_hop_path(subject)
        ret = set()
        for i in xrange(len(first_hop)):
            r1 = first_hop[i][1]
            if r1 == path[1] and len(path) == 4:
                mediate_node = first_hop[i][2]
                second_hop = DBManager.get_one_hop_path(mediate_node)
                for j in xrange(len(second_hop)):
                    if second_hop[j][1] != path[3] or second_hop[j][2] == subject:
                        continue
                    ret.add(second_hop[2])
            elif r1 == path[1]:
                ret.add(first_hop[i][2])
        return ret

    @staticmethod
    def get_property(subject, relation_name):
        # get property of given subject, for instance, get "gender" property
        if not DBManager.freebase_db:
            DBManager.freebase_db = leveldb.LevelDB(globals.config.get('LevelDB', 'freebase_db'))

        try:
            values = DBManager.freebase_db.Get(subject.encode('utf8')).decode('utf8')

            values = values.split('\t')
            ret = []
            for value in values:
                value = value.split()
                if value[0].split('.')[-1] == relation_name:
                    for v in value[1:]:
                        ret.append([subject, value[0], v])
            return ret

        except KeyError:
            return []

if __name__ == '__main__':
    globals.read_configuration('../config.cfg')
    # print DBManager.get_candidate_entities("jake'sstory", 0)
    # print len(DBManager.get_one_hop_graph('m.0f2y0'))
    print DBManager.get_subgraph('m.0f2y0')
    print DBManager.get_multiple_hop_relations('m.0f2y0')

