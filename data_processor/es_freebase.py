
import sys
sys.path.insert(0, '..')

from utils.es_helper import EsTemplate

schema = {
    'id':{
        'type': 'string',
        'index': 'not_analyzed'
    },
    'subject':{
        'type': 'string',
        'index': 'not_analyzed'
    },
    'relation':{
        'type': 'string',
        'index': 'not_analyzed'
    },
    'object':{
        'type': 'string',
        'index': 'not_analyzed'
    },
    # 'version':{
    #     'type': 'string',
    #     'index': 'not_analyzed'
    # }
}

mapping = {'properties': schema}


class EsFreebase(object):

    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 9200
        self.es = EsTemplate(self.host, self.port, 'freebase', 'triples')

    def create_freebase_store(self):
        self.es.init(mapping)

    def remove_freebase_store(self):
        self.es.clean()

    def remap_freebase_store(self, new_mapping):
        self.es.remap(new_mapping)

    def add_triples_to_store(self, filename, version):
        with open(filename) as fin:
            for line in fin:
                subject, relation, objects = line.decode('utf8').strip().split('\t')
                objects = objects.split()
                data = {}
                data['subject'] = subject
                data['relation'] = relation
                data['version'] = version
                for o in objects:
                    data['id'] = subject+relation+o
                    if self.es.exists(data['id']):
                        continue
                    data['object'] = o
                    self.es.insert(data)

    def get_triples(self, must_cond, must_not_cond=None):
        # must_cond, must_not_cond are dict
        if not must_not_cond:
            must_not_cond = []
        query = {"query": {
            "bool": {"must": [must_cond], "must_not": must_not_cond}}}
        ret = []
        for rows in self.es.search_iter(query, fields=['subject', 'relation', 'object']):
            ret.extend(rows)
        return ret

class FreebaseHelper(object):
    def __init__(self):
       self.esfreebase = EsFreebase()

    def is_mediate_relation(self, rel):
        return True

    def get_subgraph(self, mid):
        first_hops = self.esfreebase.get_triples({'subject': mid})
        subgraph = []
        for t1 in first_hops:
            rel = t1['relation']
            if self.is_mediate_relation(rel):
                subgraph.append([t1['subject'], t1['relation'], t1['object'], 0])
                second_hops = self.esfreebase.get_triples({'subject': t1['object']})
                for t2 in second_hops:
                    subgraph.append([t2['subject'], t2['relation'], t2['object'], 1])
            else:
                subgraph.append([t1['subject'], t1['relation'], t1['object'], 1])
        return subgraph
    

if __name__ == '__main__':
    esf = EsFreebase()
    # esf.create_freebase_store()
    # esf.add_triples_to_store('../../data/fb.triple.mini', '0')