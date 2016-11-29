import globals

globals.read_configuration('../config.cfg')


class MemoryFreebaseHelper:
    kb = None
    mediate_relations = None

    def __init__(self):
        self.load_kb()
        self.load_mediator_relation()

    def load_kb(self):
        if not self.kb:
            kb_filename = globals.config.get('FREEBASE', 'freebase-file')
            self.kb = dict()
            with open(kb_filename) as fin:
                for line in fin:
                    subject, relation, objects = line.decode('utf8').strip().split('\t')
                    if subject not in self.kb:
                        self.kb[subject] = dict()
                    if relation not in self.kb[subject]:
                        self.kb[subject][relation] = objects.split()

    def load_mediator_relation(self):
        if not self.mediate_relations:
            self.mediate_relations = set()
            mediator_filename = globals.config.get('FREEBASE', 'mediator-relations')
            with open(mediator_filename) as fin:
                for line in fin:
                    rel = line.decode('utf8').strip()
                    if rel.startswith('m.'):
                        continue
                    self.mediate_relations.add(rel)

    def is_mediate_relation(self, rel):
        return rel in self.mediate_relations

    def get_subgraph(self, subject):
        res = list()
        if subject not in self.kb:
            return res

        for r1, objs1 in self.kb[subject].iteritems():
            if self.is_mediate_relation(r1):
                for o1 in objs1:
                    if o1 in self.kb:
                        for r2, objs2 in self.kb[o1].iteritems():
                            for o2 in objs2:
                                if o2 != subject:
                                    res.append([[subject, r1, o1], [o1, r2, o2]])
            else:
                for o1 in objs1:
                    if o1 != subject:
                        res.append([[subject, r1, o1]])

        return res