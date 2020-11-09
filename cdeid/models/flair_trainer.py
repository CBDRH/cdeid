from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List
from pathlib import Path


class FlairTrainer:
    def __init__(self,
                 workspace,
                 train_file,
                 dev_file,
                 test_file,
                 balanced=True):
        self.workspace = workspace
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.balanced = balanced

    def get_work_dir(self):
        if self.balanced:
            return 'balanced'
        return ''

    def train(self):
        # define the columns of data files
        columns = {0: 'text', 1: 'ner'}

        # data dir
        data_dir = Path(self.workspace) / 'data' / self.get_work_dir()
        model_path = Path(self.workspace) / 'models' / self.get_work_dir()

        corpus: Corpus = ColumnCorpus(data_dir, columns,
                                      train_file=self.train_file,
                                      dev_file=self.dev_file,
                                      test_file=self.test_file)

        tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
        ]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type='ner',
                                                use_crf=True)

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(model_path,
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150)
