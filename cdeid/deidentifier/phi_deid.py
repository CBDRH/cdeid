import logging

from cdeid.deidentifier.annotator import annotate_doc
from cdeid.display.html_generator import generate_html
from cdeid.models.ensemble_model import EnsembleModel
from cdeid.utils.resources import PACKAGE_NAME
from pathlib import Path

logger = logging.getLogger(PACKAGE_NAME)


class PHIDeid:
    def __init__(self,
                 workspace,
                 deid_output_dir):
        logger.info('Loading models......')
        model_dir = Path(workspace) / 'models'
        self.model = EnsembleModel(str(model_dir / 'balanced' / 'best-model.pt'),
                                       str(model_dir / 'balanced' / 'model-best'),
                                       str(model_dir / 'balanced' / 'stanza_model.pt'),
                                       str(model_dir / 'best-model.pt'),
                                       str(model_dir / 'model-best'),
                                       str(model_dir / 'stanza_model.pt'))
        logger.info('Model loaded......')
        self.deid_output_dir = deid_output_dir
        # self.deid_file = deid_file

    def __call__(self, deid_file):
        self.deid_file = deid_file
        with open(self.deid_file, 'r') as f:
            content = f.read()
            doc, _ = self.model.predict(content)

        return doc

    def output(self, doc):
        html_file = self.deid_output_dir + '/' \
                    + str(Path(Path(self.deid_file).parts[-1]).with_suffix('.html'))
        generate_html(doc, html_file)

        annotated_file = self.deid_output_dir + '/' \
                         + str(Path(Path(self.deid_file).parts[-1]).with_suffix('.annotated.txt'))
        annotate_doc(doc, annotated_file)



