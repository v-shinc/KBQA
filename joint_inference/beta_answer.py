__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
from pipeline import Pipeline
from kb_manager.db_manager import DBManager
from beta_ranker import BetaRankerModel
import tensorflow as tf
from data_helper import DataSet
import os
import json
import globals


class BetaRanker:
    def __init__(self, dir_path):
        checkpoint_dir = os.path.abspath(os.path.join(dir_path, "checkpoints"))
        save_path = os.path.join(checkpoint_dir, "model")
        config_path = os.path.join(dir_path, 'config.json')
        parameters = json.load(open(config_path))
        parameters['reload'] = True
        parameters['load_path'] = save_path
        with tf.Graph().as_default():
            self.model = BetaRankerModel(parameters)
        self.dataset = DataSet(parameters)


    def rank_queries(self, queries):
        data = self.dataset.create_model_input(queries)
        scores, _ = self.model.predict(
            data['pattern_word_ids'],
            data['sentence_lengths'],
            None,  # TODO: support pattern char-based feature
            None,
            data['relation_ids'],
            data['relation_lengths'],
            data['mention_char_ids'],
            data['topic_char_ids'],
            data['mention_lengths'],
            data['topic_lengths'],
            data['question_word_ids'],
            data['question_lengths'],
            data['type_ids'],
            data['type_lengths'],
            data['answer_type_ids'],
            data['answer_type_weights'],
            data['qword_ids'],
            data['extras']
        )
        return dict(zip(data['hash'], scores))


class BetaAnswer(object):
    def __init__(self):
        self.pipeline = Pipeline()
        self.ranker = BetaRanker(globals.config.get('BetaAnswer', 'beta-ranker'))

    def answer(self, question):
        print '[BetaAnswer.answer]'
        question, entity_link_res, query_graphs = self.pipeline.gen_candidate_query_graph_for_prediction(question)
        for i in xrange(len(entity_link_res)):
            mid = entity_link_res[i]['topic']
            entity_link_res[i]['description'] = DBManager.get_description(mid)
            types = []
            for t in DBManager.get_notable_type(mid):
                name = DBManager.get_name(t)[0]
                if name:
                    types.append(name)
            entity_link_res[i]['notable_type'] = ' '.join(types)
            entity_link_res[i]['topic_name'] = DBManager.get_name(mid)[0]
        print ' rank query pattern'
        if len(query_graphs) == 0:
            return question, [], []
        hash_to_score = self.ranker.rank_queries(query_graphs)
        for i in range(len(query_graphs)):
            hcode = query_graphs[i]['hash']
            query_graphs[i]['score'] = float(hash_to_score.get(hcode, -2.))

        query_graphs = sorted(query_graphs, key=lambda x: x['score'], reverse=True)
        return question, entity_link_res, query_graphs





