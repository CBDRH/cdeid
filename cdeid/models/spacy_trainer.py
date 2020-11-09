import spacy
from spacy.cli import download, convert, train
from pathlib import Path
import logging

from cdeid.utils.resources import PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)


class SpacyTrainer:
    def __init__(self,
                 workspace,
                 phi_types,
                 train_file,
                 dev_file,
                 balanced=True):
        self.balanced = balanced
        self.workspace = workspace
        self.phi_types = phi_types
        self.train_file = train_file
        self.dev_file = dev_file

    def get_work_dir(self):
        if self.balanced:
            return 'balanced'
        return ''

    def convert_data(self):
        balanced = self.get_work_dir()
        convert(Path(self.workspace) / 'data' / balanced / self.train_file,
                Path(self.workspace) / 'data' / balanced,
                converter='ner')
        convert(Path(self.workspace) / 'data' / balanced / self.dev_file,
                Path(self.workspace) / 'data' / balanced,
                converter='ner')

    def customize_base_mode(self):
        if not Path(self.workspace + '/models/en_core_web_md').exists():
            # download en_core_web_lg model
            try:
                en_lg_model = spacy.load('en_core_web_lg')
            except:
                download('en_core_web_lg')
                en_lg_model = spacy.load('en_core_web_lg')

            # add customized NER types based on en_core_web_lg
            ner = en_lg_model.get_pipe('ner')
            for phi in self.phi_types:
                ner.add_label(phi)

            logger.info('Customized PHI Types added into the base model: {}'.format(str(self.phi_types)))
            en_lg_model.to_disk(Path(self.workspace) / 'models' / 'en_core_web_md')

        return self.workspace + '/models/en_core_web_md'

    def train(self):
        logger.info('Convert training data')
        self.convert_data()
        logger.info('Customize base model')
        base_model = self.customize_base_mode()
        data_dir = Path(self.workspace) / 'data' / self.get_work_dir()
        model_path = Path(self.workspace) / 'models' / self.get_work_dir()
        train_file = Path(Path(self.train_file).parts[-1]).with_suffix('.json')
        dev_file = Path(Path(self.dev_file).parts[-1]).with_suffix('.json')

        train('en', model_path, data_dir / train_file,
              data_dir / dev_file, base_model=base_model, pipeline='ner')
