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

    def scroll(self, si):
        res = self.es.scroll(scroll_id=si, scroll='10m')
        si = res['_scroll_id']
        res = [r['_source'] for r in res['hits']['hits']]
        return {'si': si, 'rs': res}

    def search(self, q, size=500, fields="*"):
        res = self.es.search(index=self.index, doc_type=self.doc_type,
                             body=q, scroll='10m', search_type='scan', _source_include=fields, size=size)
        si = res['_scroll_id']
        ct = res['hits']['total']
        return {'si': si, 'ct': ct}

    def search_iter(self, query, fields, size=500):
        res = self.search(query, size, fields)
        si = res['si']
        ci = res['ci']

        while True:
            res = self.scroll(si)
            si = res['si']
            if not res['rs']:
                break
            yield res['rs']


