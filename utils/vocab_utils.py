from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

__all__ = ["create_reverse_vocab_table", "create_vocab_table",
           "check_vocab", "load_embed_txt", "load_vocab"]

UNK_ID = 0
UNK = "<unk>"
TOS_ID = 1
TOS = "<S>"
TOE_ID = 2
TOE = "</S>"


def create_vocab_table(src_vocab_file):
    """Create vocabulary table from file"""
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    return src_vocab_table


def create_reverse_vocab_table(src_vocab_file):
    reverse_vocab_table = lookup_ops.index_to_string_table_from_file(
        src_vocab_file, default_value=UNK)
    return reverse_vocab_table


def check_vocab(vocab_file,
                out_dir,
                check_special_token=True,
                tos=None,
                toe=None,
                unk=None):
    if tf.gfile.Exists(vocab_file):
        print("# Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            if not unk: unk = UNK
            if not tos: tos = TOS
            if not toe: toe = TOE
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != tos or vocab[2] != toe:
                print("The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], unk, tos, toe))
                vocab = [unk, tos, toe] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                        tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                    vocab_file = new_vocab_file
    else:
        raise ValueError("vocab file '%s' does not exit," % vocab_file)

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def load_vocab_dict(vocab_file):
    vocab = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_id = 0
        for word in f:
            vocab[word.replace("\n", "")] = vocab_id
            vocab_id += 1

    print("vocab_size", vocab_id)
    return vocab


def load_vocab(vocab_file):
    vocab = []
    global TOS_ID
    global TOE_ID
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            word = word.replace("\n", "")
            vocab_size += 1
            vocab.append(word)
        TOS_ID = vocab.index(TOS)
        TOE_ID = vocab.index(TOE)

    print("vocab_size", vocab_size)
    return vocab, vocab_size


def load_embed_txt(embed_file):
    """Load embedding file into python dictionary
    Where embedding file should be a Glove/word2vec
    """

    emb_dict = dict()
    emb_size = 0

    is_first_line = True
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
        for line in f:
            tokens = line.rstrip().split(" ")
            if is_first_line:
                is_first_line = False
                emb_size = len(list(map(float, tokens[1:])))
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                if emb_size != len(vec):
                    print(
                        "Ignoring %s since embeding size is inconsistent." % word)
                    del emb_dict[word]
                else:
                    emb_size = len(vec)
    return emb_dict, emb_size

