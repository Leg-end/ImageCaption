from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import tensorflow as tf
from . import imc_model
from . import model_helper
import codecs
from .utils import misc_utils as utils
from .utils import caption_generator
from .utils import evaluate_utils


def get_model_creator():
    return imc_model.BaseModel


def start_sess_and_load_model(infer_model, ckpt_path):
    sess = tf.Session(
        graph=infer_model.graph, config=utils.get_config_proto())
    with infer_model.graph.as_default():
        loaded_infer_model = model_helper.load_model(
          infer_model.model, ckpt_path, sess,
          name="infer")
    return sess, loaded_infer_model


def evaluate(hparams):
    inference_inputs = hparams.inference_inputs
    eval_output_dir = hparams.eval_output_dir
    model_creator = get_model_creator()
    hparams.mode = "infer"
    infer_model = model_helper.create_infer_model(model_creator, hparams)
    sess, loaded_infer_model = start_sess_and_load_model(infer_model,
                                                         os.path.join(hparams.out_dir, "ckpt", "checkpoint"))
    output_infer = os.path.join(eval_output_dir, "eval_results.txt")
    generator = caption_generator.CaptionGenerator(loaded_infer_model,
                                                   beam_size=1,
                                                   max_caption_length=hparams.max_caption_length)
    with tf.gfile.GFile(output_infer, mode="a") as trans_f:
        trans_f.write("")
        # Ignore begin and end words.
        for k, filename in enumerate(inference_inputs):
            if k % 100 == 0:
                print(k, " epoch of evaluate")
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            #  ...........................................................................
            init_state = loaded_infer_model.eval_initial_state(sess, image_feed=image)
            captions = generator.beam_search(sess, init_state)
            for i, caption in enumerate(captions):
                capsen = caption.sentence
                end_index = capsen.index(2) if 2 in capsen else -1  # remove </S>
                sentence = [loaded_infer_model.decode(sess, w) for w in capsen[1:end_index]]
                sentence = b" ".join(sentence)
                sentence = sentence.decode("utf-8")
                trans_f.write("%s\n" % sentence)
    sess.close()
    eval_score = evaluate_utils.evaluate(output_infer, "D:/image_caption/dataset/val_captions.txt")
    print("bleu score:", eval_score)
