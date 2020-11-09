import argparse
import sys

from cdeid.deidentifier.phi_deid import PHIDeid
from cdeid.models.trainer import Trainer
from cdeid.utils.resources import PACKAGE_NAME


def get_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=PACKAGE_NAME)
    parser.add_argument('--command', type=str,
                        choices=[
                            'train',
                            'deid'], required=True)
    parser.add_argument('--data_dir', type=str, help='data sets directory')
    parser.add_argument('--train_file', type=str, default='train.bio', help='the file name of training set')
    parser.add_argument('--dev_file', type=str, default='dev.bio', help='the file name of development set')
    parser.add_argument('--test_file', type=str, default='test.bio', help='the file name of test set')
    parser.add_argument('--workspace', type=str, required=True,
                        help='the workplace which is used to store data and trained models')
    parser.add_argument('--resume_training', type=bool, default=False, help='resume the last training process')
    parser.add_argument('--phi_types', type=str, nargs='+', help='customized PHI types')

    parser.add_argument('--wordvec_file', type=str, help='wordvec files')

    #
    parser.add_argument('--deid_file', type=str, help='the file name to be de-identified')
    parser.add_argument('--deid_output_dir', type=str, help='the file directory to be de-identified')

    options = parser.parse_args(args)
    return options


def main():
    options = get_options()
    command = options.command

    if command == 'train':
        trainer = Trainer(options.data_dir,
                          options.workspace,
                          options.phi_types,
                          options.wordvec_file,
                          options.resume_training,
                          options.train_file,
                          options.dev_file,
                          options.test_file)
        trainer.train()
    elif command == 'deid':
        deider = PHIDeid(options.workspace,
                         options.deid_output_dir)
        doc = deider(options.deid_file)
        deider.output(doc)


if __name__ == '__main__':
    main()
