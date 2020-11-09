from cdeid.display.html_generator import add_html_tags_of_entities, generate_html
import logging

from cdeid.utils.resources import PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)


class Token:
    def __init__(self, text, ner_tag):
        self.text = text
        self.ner_tag = ner_tag


class Sentence:
    def __init__(self):
        self.id = 0
        self.text = ''
        self.sentence_number = 0
        self._tokens = []
        self._entities = []  # entity (start_idx, end_idx, type)
        self._ents = []  # entity (text, type)
        self._html_text = ''

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    def add_token(self, token):
        self._tokens.append(token)

    # entity (start_idx, end_idx, type)
    @property
    def entities(self):
        self._entities = self._parse_entities('index')
        return self._entities

    # entity (text, type)
    @property
    def ents(self):
        self._ents = self._parse_entities('plain')
        return self._ents

    # @property
    # def html_text(self):
    #     return self._html_text
    #
    # @html_text.setter
    # def html_text(self, html_text):
    #     self._html_text = html_text

    # @property
    # def annotated_text(self):
    #     tmp_text = self.text
    #     offset = 0
    #     for entity in self._entities:
    #         tmp_text = tmp_text[:entity[0] + offset] + '<**' + entity[2] + '**>' + tmp_text[entity[1] + offset:]
    #         offset += (len(entity[2]) + 6 - (entity[1] - entity[0]))
    #     return tmp_text

    # mode: index (start_idx, end_idx, type) , plain (text, type)
    def _parse_entities(self, mode='index'):
        entity_list = []

        current_index = 0
        entity_flag = False
        entity_type = None
        entity_tokens = []
        for i, token in enumerate(self._tokens):
            if token.ner_tag.startswith('B-'):
                entity_type = token.ner_tag[2:].strip()
                entity_flag = True
                if not (token.text.isspace() or token.text == ''):
                    entity_tokens.append(token)
            if token.ner_tag.startswith('I-') and entity_flag:
                entity_tokens.append(token)
            if (token.ner_tag == 'O' or i == len(self._tokens) - 1) and entity_flag:
                entity_flag = False

                # check if the last token is whitespace
                if len(entity_tokens) != 0 and (entity_tokens[-1].text.isspace() or entity_tokens[-1].text == ''):
                    entity_tokens.pop()
                if len(entity_tokens) == 0:
                    entity_tokens = []
                    entity_type = None
                    continue

                start_idx = self.text.index(entity_tokens[0].text, current_index)
                end_idx = self.text.index(entity_tokens[-1].text, current_index) + len(entity_tokens[-1].text)
                if mode == 'index':
                    entity_list.append((start_idx, end_idx, entity_type))
                elif mode == 'plain':
                    entity_list.append((self.text[start_idx:end_idx], entity_type))
                else:
                    raise Exception('Wrong entity mode: index or plain')
                entity_tokens = []
                entity_type = None
                current_index = end_idx

        return entity_list


class Document:
    def __init__(self):
        self.text = ''
        self._sentences = []
        self._entities = []
        self._ents = []

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sentences):
        self._sentences = sentences

    def add_sentence(self, sentence):
        self._sentences.append(sentence)

    @property
    def entities(self):
        for sent in self._sentences:
            self._entities.append(sent.entities)
        return self._entities

    @property
    def ents(self):
        for sent in self._sentences:
            self._ents.append(sent.ents)
        return self._ents

    # def generate_html_file(self, output_file):
    #     return generate_html(self, output_file=output_file)

    # def annotate_text(self, output_file):
    #     annotated_sentences = [sent.annotated_text for sent in self._sentences]
    #     annotated_text = '\n'.join(annotated_sentences)
    #
    #     with open(output_file, 'w+') as fo:
    #         fo.write(annotated_text)
    #
    #     return annotated_text
