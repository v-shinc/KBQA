import sys
sys.path.insert(0, '..')
from flask import Flask, jsonify
from flask.ext.bootstrap import Bootstrap
from flask import render_template,request
from flask import abort, redirect, url_for
import json
from tagger.predict import EntityMentionTagger

static_path = 'static'

class KBQADemo(object):

    def __init__(self, dir_path):
        self.app = Flask(__name__, template_folder=static_path, static_folder=static_path)
        self.bootstrap = Bootstrap(self.app)
        self.tagger = EntityMentionTagger(dir_path)

        @self.app.route("/kbqa")
        def main():
            return render_template('main.html', params={})

        @self.app.route('/get_entity_mention', methods=['GET', 'POST'])
        def get_entity_mention():
            print "[get_entity_mention"
            sentence = request.args.get('sentence')
            print sentence
            mention, sentence_, tag_sequence = self.tagger.tag(sentence, True)
            res = {'mention': mention, 'sentence': sentence_, 'tag_sequences': tag_sequence}
            return json.dumps(res)


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
    obj = KBQADemo(sys.argv[1])
    obj.run(7778, True)
