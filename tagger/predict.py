#coding=utf8
import sys
sys.path.insert(0, '..')
import os
import json
import re
import tensorflow as tf
from model import DeepCRF
from data_helper import DataSet
from utils.string_utils import naive_split
from corenlp_parser.local_parser import NLPParser
from kb_manager.db_manager import DBManager


def find_word(sentence, word):
    word_len = len(word)
    sentence_len = len(sentence)
    text = ' '.join(word)
    for i in range(sentence_len - word_len + 1):
        if sentence[i] == word[0] and ' '.join(sentence[i:i + word_len]) == text:
            return i
    return -1


class EntityMentionTagger(object):
    def __init__(self, dir_path):

        checkpoint_dir = os.path.abspath(os.path.join(dir_path, "checkpoints"))
        save_path = os.path.join(checkpoint_dir, "model")
        config_path = os.path.join(dir_path, 'config.json')
        parameters = json.load(open(config_path))
        with tf.Graph().as_default():
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
        self.tag_sheme = parameters['tag_scheme']
        self.use_part_of_speech = 'pos_dim' in parameters and parameters['pos_dim'] > 0
        self.dataset = DataSet(parameters)
        self.nlp_parser = NLPParser()
        self.fix = lambda x: x.replace('-LSB-', '[').replace('-RSB-', ']').replace('-LCB-', '{').replace('-RCB-', '}').replace('-LRB-', '(').replace('-RRB-', ')')

    def get_pos_tag(self, sentence):
        sentence = naive_split(sentence)
        tokens, poss = self.nlp_parser.tag_pos(' '.join(sentence))
        tokens = [self.fix(t) for t in tokens]
        return tokens, poss

    def tag(self, sentence):
        if self.use_part_of_speech:
            sentence, poss = self.get_pos_tag(sentence)
        else:
            sentence = naive_split(sentence)
            poss = None
        data = self.dataset.create_model_input(sentence, poss)
        viterbi_sequences, _ = self.model.predict(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids'],
        )
        viterbi_sequence = viterbi_sequences[0]
        seq_len = data['sentence_lengths'][0]

        words = data['words'][0][:seq_len]
        mentions, pred_tag_sequence = self.dataset.get_mention_from_words(words, viterbi_sequence)
        mention_to_likelihood = dict()
        likelihood = self.get_sequence_likelihood(data, viterbi_sequences)[0]

        for m in mentions:
            mention_to_likelihood[m] = likelihood
        res = dict()
        res['sentence'] = ' '.join(sentence)
        res['mentions'] = mention_to_likelihood
        if poss:
            res['pos'] = poss
        return res

    def tag_top2(self, sentence):
        if self.use_part_of_speech:
            sentence, poss = self.get_pos_tag(sentence)
        else:
            sentence = naive_split(sentence)
            poss = None
        data = self.dataset.create_model_input(sentence, poss)

        viterbi_sequences, scores = self.model.predict_top_k(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids'],
        )
        seq_len = data['sentence_lengths'][0]
        words = data['words'][0][:seq_len]
        mention_to_likelihood = dict()
        for k in range(2):
            if k == 1 and scores[0][1] * 1.0 / scores[0][0] < 0.95:
                break
            viterbi_sequence_ = viterbi_sequences[0][k]
            likelihood = self.get_sequence_likelihood(data, [viterbi_sequence_])[0]

            pred_entities, pred_tag_sequence = self.dataset.get_mention_from_words(words, viterbi_sequence_)
            for e in pred_entities:
                if e not in mention_to_likelihood:
                    mention_to_likelihood[e] = likelihood

        res = dict()
        res['mentions'] = mention_to_likelihood
        res['sentence'] = ' '.join(sentence)
        if poss:
            res['pos'] = poss
        return res

    def get_sequence_likelihood(self, batch_data, batch_sequence):
        batch_sequence = [self.dataset.pad_xx(s, self.dataset.tag_padding) for s in batch_sequence]
        scores = self.model.get_likelihood(
            batch_sequence,
            batch_data['sentence_lengths'],
            batch_data['word_ids'],
            batch_data['char_for_ids'],
            batch_data['char_rev_ids'],
            batch_data['word_lengths'],
            batch_data['cap_ids'],
            batch_data['pos_ids'],
        )
        return scores.tolist()

    def get_mention_likelihood(self, question, mention):
        if self.use_part_of_speech:
            sentence, poss = self.get_pos_tag(question)
        else:
            sentence = naive_split(question)
            poss = None
        mention = mention.split()
        data = self.dataset.create_model_input(sentence, poss)
        start = find_word(sentence, mention)
        end = start + len(mention)
        tag_ids = self.dataset.create_tag_sequence(start, end, len(sentence), self.tag_sheme)
        scores = self.model.get_likelihood(
            tag_ids,
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids'],
        )
        return question, scores.tolist()[0]


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

    def get_candidate_topic_entities_given_mention(self, question, mention):
        entities = DBManager.get_candidate_entities(mention, 0.1)
        candidates = []
        for e in entities:
            mid = e[0]
            entity_score = e[1]
            c = dict()
            c['mention'] = mention
            c['entity_score'] = entity_score
            c['topic'] = mid
            question, c['mention_score'] = self.entity_mention_tagger.get_mention_likelihood(question, mention)
            candidates.append(c)
        return question, candidates

    def get_candidate_topic_entities(self, sentence):
        # TODO: support gYear gDate
        """
        Returns:
            candidates: list of dict
        """
        # 需要优化： 找到所有实体及mention后， 再去统一计算mention likelihood
        res = self.entity_mention_tagger.tag(sentence)
        sentence = res['sentence']
        candidates = dict()
        for surface, likelihood in res['mentions'].items():
            # print '-' * 20
            surface_ = surface.lower().replace(' ', '')
            entity_res = DBManager.get_candidate_entities(surface_, 0.1)
            # print "key = {}, likelihood = {}, find entities {}".format(surface, likelihood, entity_res)
            for e in entity_res:
                mid = e[0]
                entity_score = e[1]
                if entity_score >= 1.1:   # alias
                    continue
                if mid not in candidates or entity_score > candidates[mid]['entity_score']:
                    candidates[mid] = dict()
                    candidates[mid]['topic'] = mid
                    candidates[mid]['mention'] = surface
                    candidates[mid]['entity_score'] = entity_score
                    candidates[mid]['mention_score'] = likelihood
        # use ngram of
        if len(candidates) == 0:
            # print '[get_candidate_topic_entities] use ngram of tagged mention'
            # all_pos = res['pos']
            for surface in res['mentions'].keys():
                surface = surface.lower().split()

                if len(surface) == 0:
                    continue
                start = find_word(sentence.split(), surface)
                # print sentence, surface, start
                if start == -1:
                    continue
                l = len(surface)
                found = False
                for j in range(l, 0, -1):
                    # if found:
                    #     break
                    for i in range(l - j + 1):
                        # if self.is_entity_occurrence(all_pos, sentence, start + i, start + i + j):
                        s = ''.join(surface[i:i + j])

                        entity_res = DBManager.get_candidate_entities(s, 0.1)
                        print surface[i:i + j], entity_res
                        for mid, entity_score in entity_res:

                            if entity_score < 1.1 and (mid not in candidates or entity_score > candidates[mid]['entity_score']):
                                candidates[mid] = dict()
                                candidates[mid]['topic'] = mid
                                candidates[mid]['mention'] = ' '.join(surface[i:i + j])
                                candidates[mid]['entity_score'] = entity_score

                                _, candidates[mid]['mention_score'] = self.entity_mention_tagger.get_mention_likelihood(sentence, ' '.join(surface[i:i + j]))
                        found = len(res) > 0
        # print '[EntityLinker.get_candidate_topic_entities] conclude'
        # for mid, info in candidates.iteritems():
        #     print mid, info
        return sentence, candidates.values()





