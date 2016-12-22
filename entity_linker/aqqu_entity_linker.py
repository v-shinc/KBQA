import sys
import sys
sys.path.insert(0, '..')
from corenlp_parser.parser import CoreNLPParser
from entity_linker import EntityLinker
from utils.string_utils import naive_split

class AqquEntityLinker:
    def __init__(self):
        self.parser = CoreNLPParser.init_from_config()
        self.entity_linker = EntityLinker.init_from_config()

    def get_candidate_topic_entities(self, question):
        question = ' '.join(naive_split(question))
        tokens = self.parser.parse(question)
        entities = self.entity_linker.identify_entities_in_tokens(tokens)
        mid_to_item = dict()

        for e in entities:
            mid = e.get_mid()
            if not mid.startswith('m.'):
                mid = e.surface_name
            if mid not in mid_to_item or mid_to_item[mid]['entity_score'] < e.surface_score:
                mid_to_item[mid] = dict()
                mid_to_item[mid]['topic'] = mid
                mid_to_item[mid]['mention'] = e.surface_name
                mid_to_item[mid]['entity_score'] = e.surface_score

        return question, mid_to_item.values()

