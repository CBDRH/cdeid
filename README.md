![cDeid](resources/cdeid_logo_2020.png)

> A framework for training de-identification models to automatically remove protected health information (PHI) from the free text.

cDeid is a customized de-identification method. The users can easily train their own de-identification Models on the 
data sets which are extracted from their own free text corpus. cDeid is based on 3 popular NLP toolkits: [spaCy][spacy], 
[Stanza][stanza] and [FLAIR][flair]. 

## Installation

```sh
pip install cdeid
```
## Usage example
We are using the [pre-trained word2vec embeddings][word2vec] released from the CoNLL 2017 Shared Task. It is important
to specify the customized PHI types in the corpus otherwise it will cause runtime error during training the models. 
### Using the Python API
#### Train the models
```python
from cdeid.models.trainer import Trainer

phi_types = ['PHONE', 'PERSON', 'ADDRESS', 'IDN', 'DOB']
nlp = Trainer("C:/data", "C:/workspace", phi_types, "C:/wordvec/English/en.vectors.xz")
nlp.train()
```
#### De-identify a sample document
```python
from cdeid.deidentifier.phi_deid import PHIDeid

deider = PHIDeid("C:/workspace", "C:/output")
doc = deider("C:/raw/example.txt")
deider.output(doc)
```
### Using the command line
#### Train the models
```sh
python cdeid --command train --workspace C:/workspace --data_dir C:/data --phi_types PHONE PERSON ADDRESS IDN DOB --wordvec_file C:/wordvec/English/en.vectors.xz
```
#### De-identify a sample document
```sh
python cdeid --command deid --workspace C:/workspace --deid_output_dir C:/output --deid_file C:/raw/example.txt
```

## Release History

* 0.1.0
    * The first release

## Contributors

Leibo Liu - initial work - [leiboliu](https://github.com/leiboliu/)

## License
[Apache License, Version 2.0](/LICENSE)

<!-- Markdown link & img dfn's -->
[spacy]: https://spacy.io/
[stanza]: https://stanfordnlp.github.io/stanza/
[flair]: https://github.com/flairNLP/flair
[word2vec]:https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y