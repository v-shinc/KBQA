from elasticsearch import Elasticsearch


class EsTemplate(object):
    def __init__(self, host, port, index, doc_type):
        self.es = Elasticsearch(hosts=[{'host': host, 'port': port}])
        self.index = index
        self.doc_type = doc_type

    def init(self, schema):
        body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0
                }
            }
        }
        # create an index in Elasticsearch,
        # index- The name of index
        # body= The configuration for the index
        self.es.indices.create(index=self.index, body=body)
        self.es.indices.put_mapping(index=self.index, doc_type=self.doc_type, body=schema)

    def remap(self, schema):
        self.es.indices.put_mapping(index=self.index, doc_type=self.doc_type, body=schema)

    def insert(self, data):
        if 'id' in data:
            self.es.index(index=self.index, doc_type=self.doc_type, id=data['id'], body=data)
        else:
            self.es.index(index=self.index, doc_type=self.doc_type, body=data)

    def get(self, id, fields='*'):
        return self.es.get(index=self.index, doc_type=self.doc_type, id=id, _source_include=fields)

    def exists(self, id):
        return self.es.exists(index=self.index, doc_type=self.doc_type, id=id)

    def clean(self):
        self.es.indices.delete_mapping(index=self.index, doc_type=self.doc_type)
        self.es.indices.delete(index=self.index)

    def search_iter(self, query, fields, size=500):
        res  = self.es.search(size=size, q=query, fields=fields)
        si = res['si']
        ci = res['ci']

        while True:
            res = self.es.scroll(si)
            if not res['rs']:
                break
            si = res['si']
            yield res['rs']


