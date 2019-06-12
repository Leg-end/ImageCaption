from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf
import numpy as np
from . import imc_model_cpu
from . import model_helper
import codecs
from .utils import misc_utils as utils
from .utils import caption_generator
from matplotlib import pyplot as plt


tf.logging.set_verbosity(tf.logging.INFO)

__all__ = ["get_model_creator", "start_sess_and_load_model",
           "infer", "plot_infer_result"]


def get_model_creator():
    return imc_model_cpu.BaseModel


def start_sess_and_load_model(infer_model, ckpt_path):
    sess = tf.Session(
        graph=infer_model.graph, config=utils.get_config_proto())
    with infer_model.graph.as_default():
        loaded_infer_model = model_helper.load_model(
          infer_model.model, ckpt_path, sess,
          name="infer")
    return sess, loaded_infer_model


def infer(hparams):
    inference_inputs = hparams.inference_inputs
    inference_output_dir = hparams.inference_output_dir
    model_creator = get_model_creator()
    infer_model = model_helper.create_infer_model(model_creator, hparams)
    sess, loaded_infer_model = start_sess_and_load_model(infer_model,
                                                         os.path.join(hparams.out_dir, "ckpt", "checkpoint"))
    output_infer = os.path.join(inference_output_dir, "img2txt_results.txt")
    summary_name = "infer_log"
    # Summary writer
    summary_path = os.path.join(inference_output_dir, summary_name)
    if not tf.gfile.Exists(summary_path):
        tf.gfile.MakeDirs(summary_path)
    summary_writer = tf.summary.FileWriter(
        summary_path, infer_model.graph)
    generator = caption_generator.CaptionGenerator(loaded_infer_model,
                                                   beam_size=hparams.beam_width,
                                                   max_caption_length=hparams.max_caption_length)
    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(output_infer, mode="a")) as trans_f:
        trans_f.write("")
        # Ignore begin and end words.
        for k, filename in enumerate(inference_inputs):
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            #  ...........................................................................
            init_state = loaded_infer_model.eval_initial_state(sess, image_feed=image)
            print(k, "infer's initial_state shape:", np.shape(init_state)) # debug
            captions = generator.beam_search(sess, init_state)
            print("Captions for image %s:" % os.path.basename(filename))
            sentences = ""
            for i, caption in enumerate(captions):
                sentence = [loaded_infer_model.decode(sess, w) for w in caption.sentence[1:caption.sentence.index(2)]]
                sentence = b" ".join(sentence)
                sentences += str(sentence)+"\n"
                trans_f.write("%s\n" % sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            plot_infer_result(filename, sentences, k)
    """else:  # do infer sequentially
        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(output_infer, mode="a")) as trans_f:
            trans_f.write("")
            for k, filename in enumerate(inference_inputs):
                with tf.gfile.GFile(filename, "rb") as f:
                    image = f.read()
                sample_words, scores, infer_summary = loaded_infer_model.decode_sequentially(sess, image)
                captions = utils.get_captions(
                    sample_words,
                    sent_id=0,
                    toe="</S>",
                    subword_option="no_spec")
                trans_f.write("%s\n" % captions)
                for i, c in enumerate(captions):
                    print("  %d) %s (p=%f)" % (i, c, scores[i]))
                plot_infer_result(filename, captions, k)"""
    summary_writer.flush()
    summary_writer.close()
    sess.close()


def plot_infer_result(imagefile, captions, order):
    plt.figure("NO."+str(order)+" Image to Caption(s)")
    image = plt.imread(imagefile)
    plt.imshow(image)
    plt.axis("off")
    plt.title(captions)
    plt.show()
