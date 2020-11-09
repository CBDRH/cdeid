from collections import Counter
import re
import spacy
import stanza
from flair.models import SequenceTagger as flair
from flair.data import Sentence as FlairSentence, build_spacy_tokenizer

from cdeid.data.data_loader import concatenate_sents
from cdeid.utils.converter import to_bio2, entity_strip
from cdeid.data.document import Document, Sentence, Token
import logging

from cdeid.utils.scorer import score_by_entity
from cdeid.utils.resources import SPACY_PRETRAINED_MODEL_LG, PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)
nlp = spacy.load(SPACY_PRETRAINED_MODEL_LG)


class EnsembleModel:
    def __init__(self, flair_model, spacy_model, stanza_model,
                 flair_model_imbalanced, spacy_model_imbalanced, stanza_model_imbalanced):
        # Stanza model object
        self.stanza_model = stanza.Pipeline(
            lang='en',
            processors={'tokenize': 'spacy'},
            ner_model_path=stanza_model,
            tokenize_no_ssplit=True
        )

        # Flair model object
        self.flair_model = flair.load(flair_model)

        # SpaCy model object
        self.spacy_model = spacy.load(spacy_model)

        # imbalanced
        # Stanza model object
        self.stanza_model_imbalanced = stanza.Pipeline(
            lang='en',
            processors={'tokenize': 'spacy'},
            ner_model_path=stanza_model_imbalanced,
            tokenize_no_ssplit=True
        )

        # Flair model object
        self.flair_model_imbalanced = flair.load(flair_model_imbalanced)

        # SpaCy model object
        self.spacy_model_imbalanced = spacy.load(spacy_model_imbalanced)

    def predict(self, text, remove_extra_whitespaces=False):
        # This flag will remove the extra whitespaces before predict
        if remove_extra_whitespaces:
            text = re.sub(' +', ' ', text)

        def majority_vote(tag1, tag2, tag3, tag4, tag5, tag6):
            # check the length
            if len(tag1) != len(tag2) or len(tag1) != len(tag3):
                raise Exception('The length of predictions different')
            # convert BIOLU and BIOES to BIO2 format
            stanza_preds_bio2 = to_bio2(tag1)
            spacy_preds_bio2 = to_bio2(tag2)

            stanza_preds_bio2_imbalanced = to_bio2(tag4)
            spacy_preds_bio2_imbalanced = to_bio2(tag5)

            final_tags = []
            tmp_tags = zip(tag3, spacy_preds_bio2, stanza_preds_bio2,
                             tag6, spacy_preds_bio2_imbalanced, stanza_preds_bio2_imbalanced)
            for token_tag in tmp_tags:
                tag = Counter(list(token_tag))
                common_tag = tag.most_common(1)[0][0]
                final_tags.append(common_tag)

            return final_tags

        def predict_line(line):
            logger.debug('Predict sentence {}'.format(line))
            sentence = Sentence()
            sentence.text = line
            if line.isspace() or line == '':
                return sentence, []
            # spacy predict
            logger.debug('Use Spacy')
            doc_spacy = self.spacy_model(line)
            preds_spacy = [token.ent_iob_
                           if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
                           for token in doc_spacy]

            logger.debug('Use Stanza')
            doc_stanza = self.stanza_model.processors['tokenize'].process(line)
            doc_stanza = self.stanza_model.processors['ner'].process(doc_stanza)
            preds_stanza = [token.ner for sent in doc_stanza.sentences for token in sent.tokens]

            # flair predict
            flair_sentence = FlairSentence(line, use_tokenizer=build_spacy_tokenizer(self.spacy_model))
            # token list by Spacy Tokenizer
            tokens = [token.text for token in doc_spacy]

            logger.debug('Use Flair')
            self.flair_model.predict(flair_sentence)

            preds_flair = [token.get_tag('ner').value for token in flair_sentence]

            # imbalanced
            # spacy predict
            logger.debug('Use Spacy')
            doc_spacy_imbalanced = self.spacy_model_imbalanced(line)
            preds_spacy_imbalanced = [token.ent_iob_
                                      if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
                                      for token in doc_spacy_imbalanced]

            logger.debug('Use Stanza')
            doc_stanza_imbalanced = self.stanza_model_imbalanced.processors['tokenize'].process(line)
            doc_stanza_imbalanced = self.stanza_model_imbalanced.processors['ner'].process(doc_stanza_imbalanced)
            preds_stanza_imbalanced = [token.ner for sent in doc_stanza_imbalanced.sentences for token in sent.tokens]

            # flair predict
            flair_sentence_imbalanced = FlairSentence(line,
                                                      use_tokenizer=build_spacy_tokenizer(self.spacy_model))
            # token list by Spacy Tokenizer
            # tokens_imbalanced = [token.text for token in doc_spacy_imbalanced]

            logger.debug('Use Flair')
            self.flair_model_imbalanced.predict(flair_sentence_imbalanced)

            preds_flair_imbalanced = [token.get_tag('ner').value for token in flair_sentence_imbalanced]

            # ner_tag = majority_vote(preds_spacy, preds_stanza, preds_flair)
            ner_tag = majority_vote(preds_stanza, preds_spacy, preds_flair,
                                    preds_stanza_imbalanced, preds_spacy_imbalanced, preds_flair_imbalanced)
            logger.debug('Get final tags')

            # remove whitespaces from begin and end of the entity
            if not remove_extra_whitespaces:
                ner_tag = entity_strip(tokens, ner_tag)

            # add token object into sentence
            for i in range(len(tokens)):
                token = Token(tokens[i], ner_tag[i])
                sentence.add_token(token)

            return sentence, list(zip(tokens, ner_tag))

        # split newlines to a list of text
        text_list = text.splitlines()
        # List of list of tuple (token, bio_tag). e.g. [[(token, bio_tag), (token, bio_tag)]]
        ner_preds = []
        # Document object to store the data
        doc = Document()
        doc.text = text
        logger.info('Start Predicting. {} lines'.format(len(text_list)))
        # preds_result = list(map(predict_line, text_list))
        preds_result = [predict_line(line) for line in text_list]
        logger.info('Complete Predicting.')
        for result in preds_result:
            doc.add_sentence(result[0])
            ner_preds.append(result[1])

        return doc, ner_preds

    def evaluate(self, text, gold_tags):
        doc_text = concatenate_sents(text)
        _, pred_tags = self.predict(doc_text)
        # pred_tags include token text. Need to transfer to bio only
        pred_ner = [[token[1] for token in sent] for sent in pred_tags]
        _, _, _, fp, fn = score_by_entity(pred_ner, gold_tags)
        # print fp and fn
        logger.info('-------------False Positive Entities---------------')
        logger.info('Entity Number: {}'.format(len(fp)))
        for fp_ent in fp:
            sent_text = text[fp_ent['sent_id']]
            logger.info('Entities: {} @ {}'.format(fp_ent, sent_text))
            # print('Entities: {} @ {}'.format(fp_ent, sent_text))
        logger.info('-------------False Negative Entities---------------')
        logger.info('Entity Number: {}'.format(len(fn)))
        for fn_ent in fn:
            sent_text = text[fn_ent['sent_id']]
            logger.info('Entities: {} @ {}'.format(fn_ent, sent_text))
