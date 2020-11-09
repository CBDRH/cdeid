import random
import logging
from pathlib import Path

from cdeid.utils.resources import PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)


def load_data(data_file):
    data = open(data_file).readlines()
    return data


def pass_bio_checking(lines):
    for i, line in enumerate(lines):
        if ('-DOCSTART-' in line) or line.strip() == '':
            continue
        # the NER tag in the last column
        ner_tag = line.split()[-1]
        if not (ner_tag.startswith('B-') or ner_tag.startswith('I-') or ner_tag.startswith('O')):
            logger.error('Input file does not have correct BIO format at line {}.'.format(i))
            return False
    return True


def parse_documents(content):
    docs = []
    doc = []
    line = []
    # convert doc to a list of lines
    for item in content:
        if '-DOCSTART-' in item:
            if len(doc) != 0:
                docs.append(doc)
                doc = []
            continue
        if item == '' or item == '\n':
            if len(line) != 0:
                doc.append(line)
                line = []
            continue
        line.append(item)

    if len(doc) != 0:
        docs.append(doc)

    return docs


# proportion will be like 1.0 or 1.5 or 2.0 or 4.0.
# Text lines with PHI is 10. 1.0 means the lines without PHI will be 10 * 1.0
def sample_data_lines(docs, proportion, output_file, file_name):
    all_lines = [line for doc in docs for line in doc]

    lines_with_phi = [line for line in all_lines if sum('B-' in token for token in line) > 0]
    lines_without_phi = [line for line in all_lines if sum('B-' in token for token in line) == 0]

    logger.info('PHI lines: {}, No PHI lines: {}'.format(len(lines_with_phi), len(lines_without_phi)))

    random.Random(2020).shuffle(lines_without_phi)
    selected_lines_without_phi = random.sample(lines_without_phi, int(len(lines_with_phi) * proportion))

    all_lines = lines_with_phi + selected_lines_without_phi

    random.Random(2020).shuffle(all_lines)
    if not Path(output_file).exists():
        Path(output_file).mkdir()
    with open(Path(output_file) / file_name, 'w', newline='\n') as f:
        for line in all_lines:
            for token in line:
                f.write(token)
            f.write('\n')
        logger.info('Data file {} created.'.format(Path(output_file) / file_name))

    # sampled_doc = []
    # for line in all_lines:
    #     sampled_doc += line
    #     sampled_doc += ['\n']

    return len(all_lines), len(lines_with_phi)
