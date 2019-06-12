"""Generally useful utility functions."""
from __future__ import print_function


import codecs
import collections
import json
import math
import os
import sys
import time
from distutils import version


import numpy as np
import six
import tensorflow as tf
import datetime

__all__ = ["check_tensorflow_version", "safe_exp",
           "print_out", "check_file_existence",
           "get_config_proto", "print_hparams",
           "load_hparams", "maybe_parse_standard_hparams",
           "save_hparams", "get_captions", "format_text",
           "format_bpe_text", "format_spm_text"]


def debug(title, context):
    debug_file = "D:/image_caption/infer_results/debug/debug_log.txt"
    np.savetxt(debug_file, context)
    now_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    debug_context = ''.join(["|\n", now_time, ":DEBUG-", title ])
    print(debug_context)
    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(debug_file, mode="a")) as debug_f:
        debug_f.write(debug_context)
   

def check_tensorflow_version():
    min_tf.version = "1.12.0"
    if(version.LooseVersion(tf.__version__) <
       version.LooseVersion(min_tf.version)):
        raise EnviromentError("Tensorflow version must >= %s" % min_tf.version)


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def print_out(s, f=None, new_line=True):
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    """if six.PY2:
        sys.stdout.write(s.encode("utf-8"))
    else:
        sys.stdout.buffer.write(s.encode("utf-8"))

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()"""


def check_file_existence(filename):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal("file %s not found.", filename)
        return False
    else:
        return True


def get_config_proto(log_device_placement=False,
                                     allow_soft_placement=True,
                                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
        if num_inter_threads:
            config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def print_hparams(hparams, skip_patterns=None, header=None):
    """Print hparams, can skip keys based on pattern."""
    if header: print_out("%s" % header)
    values = hparams.values()
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            print_out("  %s=%s" % (key, str(values[key])))


def load_hparams(model_dir):
    """Load hparams from an existing model directory."""
    hparams_file = os.path.join(model_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        print_out("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print_out("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def maybe_parse_standard_hparams(hparams, hparams_path):
    """Override hparams values with existing standard hparams config."""
    if hparams_path and tf.gfile.Exists(hparams_path):
        print_out("# Loading standard hparams from %s" % hparams_path)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_path, "rb")) as f:
            hparams.parse_json(f.read())
    return hparams


def save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    print_out("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json(indent=4, sort_keys=True))


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def get_captions(outputs, sent_id, toe, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if toe:
        toe = toe.encode("utf-8")
    # Select a sentence
    output = outputs[sent_id, :].tolist()
    # If there is an eos symbol in outputs, cut them at that point.
    if toe and toe in output:
        output = output[:output.index(toe)]
    if subword_option == "bpe":  # BPE
        captions = format_bpe_text(output)
    elif subword_option == "spm":  # SPM
        captions = format_spm_text(output)
    else:
        captions = format_text(output)
    return captions


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = b""
    if isinstance(symbols, str):
        symbols = symbols.encode()
    delimiter_len = len(delimiter)
    for symbol in symbols:
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            word += symbol[:-delimiter_len]
        else:  # end of a word
            word += symbol
            words.append(word)
            word = b""
    return b" ".join(words)


def format_spm_text(symbols):
    """Decode a text in SPM (https://github.com/google/sentencepiece) format."""
    return u"".join(format_text(symbols).decode("utf-8").split()).replace(
        u"\u2581", u" ").strip().encode("utf-8")
