import spacy
from cdeid.utils.resources import *


def ent_tuple_to_bio(sentences, bio_scheme=BIO_SCHEME_2):
    nlp = spacy.load('en_core_web_sm')
    tokens_list = []
    tokens_bio_list = []
    for record_txt, record_entities in sentences:
        entities = record_entities.get('entities', '')
        entity_strs = []
        for start, end, entity_tag in entities:
            entity_strs.append((record_txt[start:end], entity_tag))

        # replace the words with tags
        for entity_str in entity_strs:
            record_txt = record_txt.replace(entity_str[0],
                                            TAG_BEGIN + entity_str[1] + " " + entity_str[0] + ' ' + TAG_END)

        # remove the \r\n at the end of text.
        record_txt = record_txt.replace('\n', '')

        doc = nlp(record_txt)
        tokens = []
        tokens_bio = []
        entity_begin_flag = False
        entity_middle_flag = False
        entity_type = ''
        entity_token_index = 0
        same_type_entity_together = False
        previous_entity_type = ''

        for token in doc:
            if len(token.text.strip()) == 0:
                continue
            if token.text.startswith(TAG_BEGIN):
                entity_begin_flag = True
                entity_type = token.text[11:]
                if previous_entity_type != '' and previous_entity_type == entity_type:
                    same_type_entity_together = True
                continue

            if token.text.startswith(TAG_END):
                entity_begin_flag = False
                entity_middle_flag = False
                entity_type = ''
                if entity_token_index == 1 and bio_scheme == BIO_SCHEME_1:
                    tokens_bio[len(tokens_bio) - 1] = tokens_bio[len(tokens_bio) - 1].replace("B-", "I-", 1)
                entity_token_index = 0
                continue

            if entity_begin_flag and (not entity_middle_flag):
                # tokens.append(token.text)
                entity_token_index += 1
                if (same_type_entity_together and bio_scheme == BIO_SCHEME_1) or bio_scheme == BIO_SCHEME_2:
                    tokens_bio.append('B-' + entity_type)
                else:
                    tokens_bio.append('I-' + entity_type)

                entity_middle_flag = True
                previous_entity_type = entity_type

            elif entity_begin_flag and entity_middle_flag:
                # tokens.append(token.text)
                tokens_bio.append('I-' + entity_type)
                entity_token_index += 1

            elif not entity_begin_flag:
                # tokens.append(token.text)
                previous_entity_type = ''
                same_type_entity_together = False
                tokens_bio.append('O')

            tokens.append(token.text)

        # print(tokens)
        # print(tokens_bio)
        if len(tokens) != len(tokens_bio):
            raise Exception('Token length and BIO tag length are not the same.')

        # append into list
        tokens_list.append(tokens)
        tokens_bio_list.append(tokens_bio)

    return tokens_list, tokens_bio_list


# convert BIOES with list of list data structure to BIO2
def bioes_to_bio2(doc_bioes_tags):
    new_doc_tags = []
    for sent_tags in doc_bioes_tags:
        new_sent_tags = []
        for tag in sent_tags:
            if tag == 'O':
                new_sent_tags.append(tag)
            else:
                if len(tag) < 2:
                    raise Exception('Invalid BIOES tag found:{}'.format(tag))
                else:
                    if tag[:2] == 'E-':
                        new_sent_tags.append('I-' + tag[2:])
                    elif tag[:2] == 'S-':
                        new_sent_tags.append('B-' + tag[2:])
                    else:
                        new_sent_tags.append(tag)
        new_doc_tags.append((new_sent_tags))

    return new_doc_tags


def biolu_to_bio2(doc_biolu_tags):
    new_doc_tags = []
    for sent_tags in doc_biolu_tags:
        new_sent_tags = []
        for tag in sent_tags:
            if tag == 'O':
                new_sent_tags.append(tag)
            else:
                if len(tag) < 2:
                    raise Exception('Invalid BIOLU tag found:{}'.format(tag))
                else:
                    if tag[:2] == 'L-':
                        new_sent_tags.append('I-' + tag[2:])
                    elif tag[:2] == 'U-':
                        new_sent_tags.append('B-' + tag[2:])
                    else:
                        new_sent_tags.append(tag)
        new_doc_tags.append((new_sent_tags))

    return new_doc_tags


def to_bio2(sentence_bio_tags):
    new_sent_tags = []
    for tag in sentence_bio_tags:
        if tag == 'O':
            new_sent_tags.append(tag)
        else:
            if len(tag) < 2:
                raise Exception('Invalid BIO tag found:{}'.format(tag))
            else:
                if tag[:2] == 'L-' or tag[:2] == 'E-':
                    new_sent_tags.append('I-' + tag[2:])
                elif tag[:2] == 'U-' or tag[:2] == 'S-':
                    new_sent_tags.append('B-' + tag[2:])
                else:
                    new_sent_tags.append(tag)

    return new_sent_tags


# strip the start/end whitespaces of entities
def entity_strip(token_list, tag_list):
    bio_tag_list = tag_list
    for i in range(len(bio_tag_list) - 2):
        if bio_tag_list[i] == 'O' \
                and bio_tag_list[i + 1].startswith('B-') \
                and bio_tag_list[i + 2].startswith('I-'):
            if token_list[i + 1].isspace():
                bio_tag_list[i + 2] = bio_tag_list[i + 1]
                bio_tag_list[i + 1] = 'O'
        if bio_tag_list[i].startswith('I-') \
                and bio_tag_list[i + 1].startswith('I-') \
                and bio_tag_list[i + 2] == 'O':
            if token_list[i + 1].isspace():
                bio_tag_list[i + 1] = 'O'
    return bio_tag_list
