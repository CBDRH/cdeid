# Load bio2 format to Document object
# Note: the tokenizer must use spacy pipeline
# file format: columns split by whitespace
# need to pass the indexes for text and ner columns


def load_doc(data_file):
    with open(data_file, 'r') as f:
        sents = []
        sent_text = []
        ner_tags = []
        sent_ner_tags = []
        for i, line in enumerate(f):
            if line == '\n' and len(sent_text) != 0:
                sents.append(sent_text)
                sent_text = []
                ner_tags.append(sent_ner_tags)
                sent_ner_tags = []
            cols = line.split()
            if len(cols) < 2:
                continue
            sent_text.append(cols[0])
            sent_ner_tags.append(cols[1].strip())

        # doc_text = concatenate_sents(sents)
        # print(doc_text)
        return sents, ner_tags


def concatenate_sents(sentences):
    # sentences is a list of lists of tokens
    sentences = [' '.join(sent) for sent in sentences]
    doc = '\n'.join(sentences)
    return doc

def concatenate_sents_to_one(sentences):
    # sentences is a list of lists of tokens
    sentences = [' '.join(sent) for sent in sentences]
    doc = ' '.join(sentences)
    return doc
