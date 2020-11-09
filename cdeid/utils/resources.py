PACKAGE_NAME = 'cdeid'
SPACY_PRETRAINED_MODEL_LG = 'en_core_web_lg'
# SPACY_PRETRAINED_MODEL_SM = 'en_core_web_sm'
PROGRESS_STATUS = {
    1: 'prepare data sets',
    2: 'train spacy on balanced sets',
    3: 'train stanza on balanced sets',
    4: 'train flair on balanced sets',
    5: 'train spacy on imbalanced sets',
    6: 'train stanza on imbalanced sets',
    7: 'train flair on imbalanced sets',
    8: 'ensemble models',
    9: 'previous training process completed.'
}

TAG_BEGIN = 'TAGTAGBEGIN'
TAG_END = 'TAGTAGEND'
BIO_SCHEME_1 = 'BIO1'
BIO_SCHEME_2 = 'BIO2'

tag_1 = '<span class="selWords">'
tag_2 = '<span class='
tag_3 = 'tag">'
tag_4 = '</span></span>'