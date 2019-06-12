"""Evaluate the model.

This script should be run concurrently with training so that summaries show up

in TensorBoard.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

import tensorflow as tf

from . import bleu_utils as bleu


tf.logging.set_verbosity(tf.logging.INFO)


def evaluate(ref_file, trans_file, subword_option=None, max_order=4):
    evaluation_score = _bleu(ref_file, trans_file,
                             subword_option=subword_option, max_order=max_order)
    return evaluation_score


def _bleu(ref_file, trans_file, subword_option=None, max_order=4):
    """Compute BLEU scores and handling BPE."""
    smooth = False
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)
    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))
    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _clean(sentence, subword_option=None):
    sentence = sentence.strip()
    return sentence
