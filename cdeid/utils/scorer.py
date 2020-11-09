from collections import Counter
import logging

from cdeid.utils.resources import PACKAGE_NAME

logger = logging.getLogger(PACKAGE_NAME)

# This function is modified from the function of stanza/models/ner/scorer.py
# This function to calculate the separate metrics of each entity.
#
# Copyright 2019 The Board of Trustees of The Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def score_by_entity(pred_tag_sequences, gold_tag_sequences, verbose=True):
    assert (len(gold_tag_sequences) == len(pred_tag_sequences)), \
        "Number of predicted tag sequences does not match gold sequences."

    def decode_all(tag_sequences):
        # decode from all sequences, each sequence with a unique id
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bio2(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents

    gold_ents = decode_all(gold_tag_sequences)
    pred_ents = decode_all(pred_tag_sequences)

    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()

    # Added. Store all the entities
    entities = set()

    # records the details of fp and fn
    fp = []
    fn = []
    for p in pred_ents:
        if p not in gold_ents:
            fp.append(p)
    for p in gold_ents:
        if p not in pred_ents:
            fn.append(p)

    logger.info('Predict entities in total: {}'.format(len(pred_ents)))
    logger.info('Gold entities in total: {}'.format(len(gold_ents)))
    logger.info('False Positive: {}'.format(len(fp)))
    logger.info('False Negative: {}'.format(len(fn)))

    for p in pred_ents:
        guessed_by_type[p['type']] += 1
        if p in gold_ents:
            correct_by_type[p['type']] += 1
    for g in gold_ents:
        gold_by_type[g['type']] += 1
        entities.add(g['type'])

    prec_micro = 0.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)

    if verbose:
        logger.info("Prec.\tRec.\tF1")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format(prec_micro * 100, rec_micro * 100, f_micro * 100))

    # metrics for entities
    if verbose:
        logger.info("Entity\tPrec.\tRec.\tF1")
        for entity in entities:
            prec_ent = 0.0
            if guessed_by_type[entity] > 0:
                prec_ent = correct_by_type[entity] * 1.0 / guessed_by_type[entity]
            rec_ent = 0.0
            if gold_by_type[entity] > 0:
                rec_ent = correct_by_type[entity] * 1.0 / gold_by_type[entity]
            f_ent = 0.0
            if prec_ent + rec_ent > 0:
                f_ent = 2.0 * prec_ent * rec_ent / (prec_ent + rec_ent)

            logger.info("{}\t{:.2f}\t{:.2f}\t{:.2f}".format(entity, prec_ent * 100, rec_ent * 100, f_ent * 100))

    return prec_micro, rec_micro, f_micro, fp, fn


# This function is modified from the function of stanza/models/ner/utils.py
# This function to decode all the entities in the sentence tags
#
# Copyright 2019 The Board of Trustees of The Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def decode_from_bio2(tags):
    res = []
    ent_idxs = []
    cur_type = None

    def flush():
        if len(ent_idxs) > 0:
            res.append({
                'start': ent_idxs[0],
                'end': ent_idxs[-1],
                'type': cur_type})

    for idx, tag in enumerate(tags):
        if tag is None:
            tag = 'O'
        if tag == 'O':
            flush()
            ent_idxs.clear()
        elif tag.startswith('B-'):  # start of new ent
            flush()
            ent_idxs.clear()
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('I-'):  # continue last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]

    # flush after whole sentence
    flush()
    return res
