def annotate_sent(sent):
    tmp_text = sent.text
    offset = 0
    for entity in sent.entities:
        tmp_text = tmp_text[:entity[0] + offset] + '<**' + entity[2] + '**>' + tmp_text[entity[1] + offset:]
        offset += (len(entity[2]) + 6 - (entity[1] - entity[0]))
    return tmp_text


def annotate_doc(doc, output_file):
    annotated_sentences = [annotate_sent(sent) for sent in doc.sentences]
    annotated_text = '\n'.join(annotated_sentences)
    with open(output_file, 'w+') as fo:
        fo.write(annotated_text)
