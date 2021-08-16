from stanza.utils.prepare_ner_data import load_conll03
from pathlib import Path
import json
import logging
import cdeid.models.stanza_trainer_agent as stanza_agent
from cdeid.utils.resources import PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)


def bio_to_json(data_dir, file_name):
    sentences = load_conll03(Path(data_dir) / file_name)
    logger.info("{} examples loaded from {}".format(len(sentences), Path(data_dir) / file_name))

    document = []
    for (words, tags) in sentences:
        sent = []
        for w, t in zip(words, tags):
            sent += [{'text': w, 'ner': t}]
        document += [sent]

    output_file_name = 'stanza_' + str(Path(Path(file_name).parts[-1]).with_suffix('.json'))
    with open(Path(data_dir) / output_file_name, 'w') as outfile:
        json.dump(document, outfile)
    logger.info("Generated json file {}.".format(Path(data_dir) / output_file_name))
    return output_file_name


class StanzaTrainer:
    def __init__(self,
                 workspace,
                 train_file,
                 dev_file,
                 wordvec_file,
                 balanced=True):
        self.workspace = workspace
        self.train_file = train_file
        self.dev_file = dev_file
        self.wordvec_file = wordvec_file
        self.balanced = balanced

    def get_work_dir(self):
        if self.balanced:
            return 'balanced'
        return ''

    def convert_data(self):
        balanced = self.get_work_dir()
        data_dir = Path(self.workspace) / 'data' / balanced

        train_file_name = bio_to_json(data_dir, self.train_file)
        dev_file_name = bio_to_json(data_dir, self.dev_file)
        return train_file_name, dev_file_name

    def download_wordvec(self):
        # download and save wordvec in .cdeid directory in user home for the first time or importlib_resources package
        # wordvec_dir = Path.home() / '.' / PACKAGE_NAME
        # check if wordvec file already in the folder of word2vec
        # if not, download from a github link
        # if yes, return the full path of the en.vectors.xz
        return self.wordvec_file

    def train(self):
        logger.info('Convert training data')
        train_file_name, dev_file_name = self.convert_data()
        logger.info('Download pre-trained word2vec embeddings of CoNLL2017')
        wordvec_path = self.download_wordvec()
        data_dir = Path(self.workspace) / 'data' / self.get_work_dir()
        model_path = Path(self.workspace) / 'models' / self.get_work_dir()

        args = ['--train_file', str(data_dir / train_file_name), '--eval_file', str(data_dir / dev_file_name),
                '--lang', 'en', '--mode', 'train', '--wordvec_file', wordvec_path, '--save_dir', str(model_path),
                '--save_name', 'stanza_model.pt', '--batch_size', '16', '--lr', '0.1', '--dropout', '0.5']

        stanza_agent.main(args)
