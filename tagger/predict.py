import sys
sys.path.insert(0, '..')
import os
import json
import re
from model import DeepCRF
from data_helper import DataSet
from string_utils import naive_split
from corenlp_parser.local_parser import NLPParser
from utils.db_manager import DBManager


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
            parameters['pos_dim'],
            save_path,
            parameters['num_word'],
            parameters['num_char'],
            parameters['num_cap'],
            parameters['num_pos'],
            parameters['num_tag']
        )
        self.dataset = DataSet(parameters)
        self.nlp_parser = NLPParser()
        self.fix = lambda x: x.replace('-LSB-', '[').replace('-RSB-', ']').replace('-LCB-', '{').replace('-RCB-', '}').replace('-LRB-', '(').replace('-RRB-', ')')

    def get_pos_tag(self, sentence):
        sentence = naive_split(sentence)
        tokens, poss = self.nlp_parser.tag_pos(' '.join(sentence))
        tokens = [self.fix(t) for t in tokens]
        return tokens, poss

    def tag(self, sentence):
        sentence, poss = self.get_pos_tag(sentence)
        data = self.dataset.create_model_input(sentence, poss)
        viterbi_sequences = self.model.predict(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids']
        )
        viterbi_sequence = viterbi_sequences[0]
        seq_len = data['sentence_lengths'][0]

        words = data['words'][0][:seq_len]
        pred_entities, pred_tag_sequence = self.dataset.get_named_entity_from_words(words, viterbi_sequence)
        res = {}
        res['mentions'] = pred_entities
        res['sentence'] = sentence
        res['pos'] = poss
        return res

    def tag_top2(self, sentence):
        sentence, poss  = self.get_pos_tag(sentence)
        data = self.dataset.create_model_input(sentence, poss)
        viterbi_sequences, scores = self.model.predict_topk(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids']
        )

        seq_len = data['sentence_lengths'][0]
        words = data['words'][0][:seq_len]

        all_pred_entities = set()

        for k in range(2):
            if k == 1 and scores[0][1] * 1.0 / scores[0][0] < 0.95:
                break
            viterbi_sequence_ = viterbi_sequences[0][k]
            pred_entities, pred_tag_sequence = self.dataset.get_named_entity_from_words(words, viterbi_sequence_)
            all_pred_entities.update(pred_entities)
        res = {}
        res['mentions'] = all_pred_entities
        res['sentence'] = sentence
        res['pos'] = poss
        return res

class EntityLinker(object):
    def __init__(self, entity_mention_tagger):
        self.entity_mention_tagger = entity_mention_tagger

        self.valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
                                      r'RB|CC|NNP.?|NN.?|JJ.?|CD|DT|MD|'
                                      r'POS)+$')

        self.ignore = {'are', 'is', 'were', 'was', 'be', 'of', 'the', 'and', 'or', 'a', 'an'}

    def is_entity_occurrence(self, all_pos, all_token, start, end):
        # Concatenate POS-tags
        token_list = all_token[start:end]
        pos_list = all_pos[start:end]
        pos_str = ''.join(pos_list)
        # Check if all tokens are in the ignore list.
        # For length 1 only allows nouns
        if all((t in self.ignore for t in token_list)):
            return False

        if len(pos_list) == 1 and (pos_list[0].startswith('N') or pos_list[0].startswith('JJ')) \
                or len(pos_list) > 1 and self.valid_entity_tag.match(pos_str):
            if len(pos_list) == 1:
                if pos_list[0].startswith('NNP') and start > 0 and all_pos[start - 1].startswith('NNP'):
                    return False
                elif pos_list[-1].startswith('NNP') and end < len(all_pos) and all_pos[end].startswith('NNP'):
                    return False
            return True
        return False

    def find_word(self, sentence, word):
        word_len = len(word)
        sentence_len = len(sentence)
        text = ' '.join(word)
        for i in range(sentence_len - word_len + 1):
            if sentence[i] == word[0] and ' '.join(sentence[i:i + word_len]) == text:
                return i
        return -1

    def get_candidate_topic_entities(self, sentence):
        res = self.entity_mention_tagger.tag_top2(sentence)
        candidate_to_score = dict()
        candidate_to_mention = dict()
        for surface in res['mentions']:
            surface_ = surface.lower().replace(' ', '')
            res = DBManager.get_candidate_entities(surface_, 0.1)
            for e in res:
                if e[1] >= 1.1:
                    continue
                if e[0] not in candidate_to_score or e[1] > candidate_to_score[e[0]]:
                    candidate_to_score[e[0]] = e[1]
                    candidate_to_mention[e[0]] = surface

        # use ngram of
        if len(candidate_to_score) == 0:
            all_pos = res['pos']
            for surface in res['mentions']:
                surface = surface.lower().split()
                if len(surface) == 0:
                    continue
                start = self.find_word(sentence, surface)
                if start == -1:
                    continue
                l = len(surface)
                found = False
                for j in range(l, 0, -1):
                    # if found:
                    #     break
                    for i in range(l - j + 1):
                        if self.is_entity_occurrence(all_pos, sentence, start + i, start + i + j):
                            s = ''.join(surface[i:i + j])
                            res = DBManager.get_candidate_entities(s, 0.1)
                            for e in res:
                                if e[1] < 1.1 and (e[0] not in candidate_to_score or e[1] > candidate_to_score[e[0]]):
                                    candidate_to_score[e[0]] = e[1]
                                    candidate_to_mention[e[0]] = ' '.join(surface[i:i + j])
                            found = len(res) > 0
        return candidate_to_mention, candidate_to_score





