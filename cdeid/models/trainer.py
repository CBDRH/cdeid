# ---------------------------------------------------------------
# Train models using three NLP toolkits: spaCy, Stanza and FLAIR
#
# (C) 2020 Leibo Liu, Sydney NSW, Australia
# Released under Apache License 2.0
# ---------------------------------------------------------------
import logging
from pathlib import Path
from shutil import copyfile

from cdeid.data.data_loader import load_doc
from cdeid.models.ensemble_model import EnsembleModel
from cdeid.models.flair_trainer import FlairTrainer
from cdeid.models.spacy_trainer import SpacyTrainer
from cdeid.models.stanza_trainer import StanzaTrainer
from cdeid.data.data_operator import pass_bio_checking, sample_data_lines, parse_documents, load_data
from cdeid.utils.resources import PROGRESS_STATUS, PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)


class Trainer:
    def __init__(self,
                 data_dir,
                 workspace,
                 phi_types,
                 wordvec_file,
                 resume_training=False,
                 train_file='train.bio',
                 dev_file='dev.bio',
                 test_file='test.bio'
                 ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise Exception('Data path does not exist: {}'.format(data_dir))

        self.workspace = workspace
        self.phi_types = phi_types
        self.wordvec_file = wordvec_file

        self.resume = resume_training

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

    def train(self):
        """Train the models using the provided corpus

        Load the corpus of training and development and train models on balanced and
        imbalanced data sets.

        Args:

        Returns:
            no value
        """
        # prepare workspace
        self.prepare_workspace()

        # 0. check the status of training in the .status file in the workspace
        status_flags = [1] * 8

        if not self.resume:
            logger.info('A new training process is starting')
        else:
            status = self.check_train_status()
            if status - 1:
                for i in range(0, status - 1):
                    status_flags[i] = 0

            logger.info('Resume the previous training process at the step {} - {}'
                        .format(status, PROGRESS_STATUS[status]))

        # Process training
        if status_flags[0]:
            self.prepare_datasets()

        if status_flags[1]:
            self.train_spacy_balanced()

        if status_flags[2]:
            self.train_stanza_balanced()

        if status_flags[3]:
            self.train_flair_balanced()

        if status_flags[4]:
            self.train_spacy_imbalanced()

        if status_flags[5]:
            self.train_stanza_imbalanced()

        if status_flags[6]:
            self.train_flair_imbalanced()

        if status_flags[7]:
            self.ensemble_models()

    # check train status. Status stored in .status file in workspace with a status number
    def check_train_status(self):
        status_file = open(Path(self.workspace) / '.status', 'r')
        status = status_file.readline()
        status_file.close()
        if len(status.strip()) == 0:
            return 1
        return int(status)

    # 1-prepare data sets
    # 2-train spacy on balanced sets
    # 3-train stanza on balanced sets
    # 4-train flair on balanced sets
    # 5-train spacy on imbalanced sets
    # 6-train stanza on imbalanced sets
    # 7-train flair on imbalanced sets
    # 8-ensemble model
    def update_train_status(self, status):
        status_file = open(Path(self.workspace) / '.status', 'w')
        status_file.write(str(status))
        status_file.close()

    # 0-prepare workspace directory
    def prepare_workspace(self):
        workspace_dir = Path(self.workspace)
        if not workspace_dir.exists():
            workspace_dir.mkdir()

        models_dir = Path(self.workspace) / 'models'
        if not models_dir.exists():
            models_dir.mkdir()

        data_dir = Path(self.workspace) / 'data'
        if not data_dir.exists():
            data_dir.mkdir()

        status_file = Path(self.workspace) / '.status'
        if not status_file.is_file():
            status_file.touch()

        logger.info('workspace prepared: models and data directories are ready')

    # 1-prepare balanced and imbalanced training and dev data sets
    def prepare_datasets(self):
        self.update_train_status(1)
        logger.info('Training step 1 - prepare data sets - started')
        data_dir = Path(self.workspace) / 'data'
        # if not data_dir.exists():
        #     data_dir.mkdir()
        # copy data to workspace/data directory
        copyfile(self.data_dir / self.train_file, data_dir / self.train_file)
        copyfile(self.data_dir / self.dev_file, data_dir / self.dev_file)
        copyfile(self.data_dir / self.test_file, data_dir / self.test_file)
        # load data from files
        train = load_data(data_dir / self.train_file)
        if not pass_bio_checking(train):
            raise Exception('Please check the BIO format in the file {}'.format(self.train_file))

        dev = load_data(data_dir / self.dev_file)
        if not pass_bio_checking(dev):
            raise Exception('Please check the BIO format in the file {}'.format(self.dev_file))

        # parse data to list of lists for sampling the balanced data sets
        train_doc = parse_documents(train)
        logger.info('Training set: {} lines'.format(sum([len(doc) for doc in train_doc])))
        # generate balanced data sets
        train_balanced_size, _ = sample_data_lines(train_doc, 1.0, data_dir / 'balanced', self.train_file)
        logger.info('Training balanced set: {} lines'.format(train_balanced_size))

        dev_doc = parse_documents(dev)
        logger.info('Dev set: {} lines'.format(sum([len(doc) for doc in dev_doc])))
        dev_balanced_size, _ = sample_data_lines(dev_doc, 1.0, data_dir / 'balanced', self.dev_file)
        logger.info('Dev balanced set: {} lines'.format(dev_balanced_size))

        copyfile(self.data_dir / self.test_file, data_dir / 'balanced' / self.test_file)

        logger.info('Training step 1 - prepare data sets - completed')

    # 2-train spaCy model on balanced training and dev data sets
    def train_spacy_balanced(self):
        self.update_train_status(2)
        logger.info('Training step 2 - train spacy on balanced sets - started')
        spacy_trainer = SpacyTrainer(self.workspace, self.phi_types, self.train_file, self.dev_file)
        spacy_trainer.train()
        logger.info('Training step 2 - train spacy on balanced sets - completed')

    # 3-train Stanza model on balanced training and dev data sets
    def train_stanza_balanced(self):
        self.update_train_status(3)
        logger.info('Training step 3 - train stanza on balanced sets - started')
        stanza_trainer = StanzaTrainer(self.workspace, self.train_file, self.dev_file, self.wordvec_file)
        stanza_trainer.train()
        logger.info('Training step 3 - train stanza on balanced sets - completed')

    # 4-train FLAIR model on balanced training and dev data sets
    def train_flair_balanced(self):
        self.update_train_status(4)
        logger.info('Training step 4 - train flair on balanced sets - started')
        flair_trainer = FlairTrainer(self.workspace, self.train_file, self.dev_file, self.test_file)
        flair_trainer.train()
        logger.info('Training step 4 - train flair on balanced sets - completed')

    # 5-train spaCy model on imbalanced training and dev data sets
    def train_spacy_imbalanced(self):
        self.update_train_status(5)
        logger.info('Training step 5 - train spacy on imbalanced sets - started')
        spacy_trainer = SpacyTrainer(self.workspace, self.phi_types, self.train_file, self.dev_file, balanced=False)
        spacy_trainer.train()
        logger.info('Training step 5 - train spacy on imbalanced sets - completed')

    # 6-train Stanza model on imbalanced training and dev data sets
    def train_stanza_imbalanced(self):
        self.update_train_status(6)
        logger.info('Training step 6 - train stanza on imbalanced sets - started')
        stanza_trainer = StanzaTrainer(self.workspace,
                                       self.train_file, self.dev_file, self.wordvec_file, balanced=False)
        stanza_trainer.train()
        logger.info('Training step 6 - train stanza on imbalanced sets - completed')

    # 7-train FLAIR model on imbalanced training and dev data sets
    def train_flair_imbalanced(self):
        self.update_train_status(7)
        logger.info('Training step 7 - train flair on imbalanced sets - started')
        flair_trainer = FlairTrainer(self.workspace, self.train_file, self.dev_file, self.test_file, balanced=False)
        flair_trainer.train()
        logger.info('Training step 7 - train flair on imbalanced sets - completed')

    # 8-ensemble models
    def ensemble_models(self):
        self.update_train_status(8)
        logger.info('Training step 8 - ensemble model - started')
        logger.info('Evaluate the ensemble model on the test set')
        model_dir = Path(self.workspace) / 'models'
        test_set_path = Path(self.workspace) / 'data' / self.test_file
        ensemble_model = EnsembleModel(str(model_dir / 'balanced' / 'best-model.pt'),
                                       str(model_dir / 'balanced' / 'model-best'),
                                       str(model_dir / 'balanced' / 'stanza_model.pt'),
                                       str(model_dir / 'best-model.pt'),
                                       str(model_dir / 'model-best'),
                                       str(model_dir / 'stanza_model.pt')
                                       )
        doc_text, gold_tags = load_doc(test_set_path)
        ensemble_model.evaluate(doc_text, gold_tags)
        # ensemble_model.evaluate_with_batch(doc_text, gold_tags, batch_size=64)

        logger.info('Training step 8 - ensemble model - completed')
        # mark the training process as completed
        self.update_train_status(9)
