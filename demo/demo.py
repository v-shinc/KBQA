import sys
sys.path.insert(0, '..')
from flask import Flask
from flask.ext.bootstrap import Bootstrap
from flask import render_template,request
from utils.db_manager import DBManager
import json
import globals

static_path = 'static'

class KBQADemo(object):

    def __init__(self, dir_path):
        self.app = Flask(__name__, template_folder=static_path, static_folder=static_path)
        self.bootstrap = Bootstrap(self.app)
        # self.tagger = EntityMentionTagger(dir_path)

        @self.app.route("/kbqa")
        def main():
            return render_template('main.html', params={})

        @self.app.route('/get_entity_mention', methods=['GET', 'POST'])
        def get_entity_mention():
            print "[get_entity_mention]"
            sentence = request.args.get('sentence')
            print sentence
            mention, sentence_, tag_sequence = self.tagger.tag(sentence, True)
            res = {'mention': mention, 'sentence': sentence_, 'tag_sequences': tag_sequence}
            return json.dumps(res)

        @self.app.route('/mid_to_name', methods=['GET', 'POST'])
        def get_name_by_mid():
            mid = request.args.get('mid')
            print "[get_name_by_mid]", mid

            name, name_info = DBManager.get_name(mid)
            print name, name_info
            aliases, alias_info = DBManager.get_alias(mid)
            res = {'name': name if name else name_info,
                   'alias': '<br>'.join(aliases) if aliases else alias_info}
            return json.dumps(res)
        @self.app.route('/surface_to_mids', methods=['GET', 'POST'])
        def get_mids_by_surface():
            surface = request.args.get('surface')
            print '[get_mid_by_surface]'

            mids = DBManager.get_candidate_entities(surface, 0.1)
            print mids
            res = {'candidates': '<br>'.join('%s %s' % (m[0], m[1]) for m in mids)}
            return json.dumps(res)
        @self.app.route('/get_subgraph', methods=['GET', 'POST'])
        def get_subgraph():
            mid = request.args.get('mid')
            data = []
            data.append({'name': "chensn"})
        # @self.app.route('/t/get_title_nns/', methods=['GET', 'POST'])
        # def get_title_nns():
        #     # query = request.args.get('query', '', type=str)
        #     title = request.args.get('title')
        #     # print query
        #     # print request.args
        #     # nns : [(title, score),...,]
        #     nns = self.ns.get_k_nearest_title(title, 20)
        #
        #     # for t, s in nns:
        #     #     print t, s
        #     nns = [{'title': t, 'score': s} for t, s in nns]
        #     return json.dumps(nns)
        #
        # @self.app.route('/t/get_answers/', methods=['GET', 'POST'])
        # def get_answers():
        #     query = request.args.get('query')
        #     print query
        #     nns = self.ns.get_answers(query, 20)
        #
        #     print '[in get_answers]'
        #     for q, s in nns:
        #         print q, s
        #     nns = [{'query': q, 'score': s} for q, s in nns]
        #     return json.dumps(nns)

    def run(self, port,debug=False):
        self.app.run(host='0.0.0.0',port=port, debug=debug)

if __name__ == '__main__':
    globals.read_configuration('../config.cfg')
    obj = KBQADemo(sys.argv[1])
    obj.run(7778, True)
