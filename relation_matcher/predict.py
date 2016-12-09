from model import RelationMatcherModel
from data_helper import DataSet
import os
import json
import tensorflow as tf


class RelationMatcher:
    def __init__(self, dir_path):
        checkpoint_dir = os.path.abspath(os.path.join(dir_path, "checkpoints"))
        save_path = os.path.join(checkpoint_dir, "model")
        config_path = os.path.join(dir_path, 'config.json')
        parameters = json.load(open(config_path))

        parameters['reload'] = True
        parameters['load_path'] = save_path
        with tf.Graph().as_default():
            self.model = RelationMatcherModel(parameters)
        self.dataset = DataSet(parameters)

    def get_match_score(self, pattern, relation):
        data = self.dataset.create_model_input([pattern], [relation])
        scores = self.model.predict(
            data['word_ids'],
            data['sentence_lengths'],
            data['char_ids'],
            data['word_lengths'],
            data['relation_ids'],
        )
        return scores[0]

    def get_batch_match_score(self, patterns, relations):
        # TODO: if number of relation is big, compute score in batches
        data = self.dataset.create_model_input(patterns, relations)
        scores, pattern_repr, relation_repr = self.model.predict(
            data['word_ids'],
            data['sentence_lengths'],
            data['char_ids'],
            data['word_lengths'],
            data['relation_ids'],
            include_repr=True
        )
        return scores, pattern_repr, relation_repr

