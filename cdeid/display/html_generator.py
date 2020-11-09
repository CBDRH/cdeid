from mako.template import Template
from pathlib import Path

HTML_TAG_1 = '<span class="entity '
HTML_TAG_2 = '"><span class="words">'
HTML_TAG_3 = '</span><span class="ner">'
HTML_TAG_4 = '</span></span>'


def add_html_tags_of_entities(text, entities):
    offset = 0
    for entity in entities:
        tag_before = HTML_TAG_1 + entity[2].lower() + HTML_TAG_2
        tag_after = HTML_TAG_3 + entity[2].upper() + HTML_TAG_4

        text = text[:(entity[0] + offset)] \
               + tag_before + text[(entity[0] + offset):(entity[1] + offset)] \
               + tag_after + text[(entity[1] + offset):]

        offset += (len(tag_before) + len(tag_after))

    return text


def generate_html(doc, output_file='./display.html', template='display_template.html'):
    template_file = str(Path(__file__).parent.absolute() / template)
    temp = Template(filename=template_file)
    tagged_lines = []
    for sent in doc.sentences:
        tagged_lines.append(add_html_tags_of_entities(sent.text, sent.entities))

    html_content = temp.render(doc=tagged_lines)

    with open(output_file, 'w+') as fo:
        fo.write(html_content)
