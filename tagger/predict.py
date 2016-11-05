import os
import json
from model import DeepCRF
from data_helper import DataSet

class EntityMentionTagger(object):
    def __init__(self, dir_path):

        checkpoint_dir = os.path.abspath(os.path.join(dir_path, "checkpoints"))
        save_path = os.path.join(checkpoint_dir, "model")
        config_path = os.path.join(dir_path, 'config.json')
        parameters = json.load(open(config_path))
        self.model = DeepCRF(
            parameters['max_sentence_len'],
            parameters['max_word_len'],
            parameters['char_dim'],
            parameters['char_rnn_dim'],
            parameters['char_bidirect'] == 1,
            parameters['word_dim'],
            parameters['word_rnn_dim'],
            parameters['word_bidirect'] == 1,
            parameters['cap_dim'],
            save_path,
            parameters['num_word'],
            parameters['num_char'],
            parameters['num_cap'],
            parameters['num_tag']
        )
        self.dataset = DataSet(parameters['fn_word'], parameters['fn_char'], parameters)

    def tag(self, sentence, include_info=False):
        data = self.dataset.create_model_input(sentence)
        viterbi_sequences = self.model.predict(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
        )
        viterbi_sequence = viterbi_sequences[0]
        seq_len = data['sentence_lengths'][0]

        words = data['words'][0][:seq_len]
        pred_entities, pred_tag_sequence = self.dataset.get_named_entity_from_words(words, viterbi_sequence)
        if include_info:
            return pred_entities, words, pred_tag_sequence
        else:
            return pred_entities