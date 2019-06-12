from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
import os

from . import run_train
from . import run_inference
from . import run_evaluate
from .utils import misc_utils as utils
from .utils import standard_hparams_utils


def run_main(hparams, train_fn, infer_fn, eval_fn):

    # GPU device
    print(
        "# Devices visible to TensorFlow: %s" % repr(tf.Session().list_devices()))

    # Random
    random_seed = hparams.random_seed
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d" % random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Model output directory
    out_dir = hparams.out_dir
    if out_dir and not tf.gfile.Exists(out_dir):
        utils.print_out("# Creating output directory %s ..." % out_dir)
        tf.gfile.MakeDirs(out_dir)
    mode = hparams.mode
    if hparams.inference_inputs and mode == "infer":
        inference_output_dir = hparams.inference_output_dir
        if not utils.check_file_existence(inference_output_dir):
            print("# Creating output directory %s ..." % inference_output_dir)
            tf.gfile.MakeDirs(inference_output_dir)

        inference_inputs = hparams.inference_inputs
        inputs_len = len(inference_inputs)

        if inputs_len == 1:  # single file
            hparams.inference_indices = [0]
            if not utils.check_file_existence(inference_inputs[0]):
                raise ValueError("Can not find file %s" % hparams.inference_inputs)
        elif inputs_len > 1:  # multi files
            for i, _ in enumerate(inference_inputs):
                hparams.inference_indices.append(i)  # store decode_id

        print(hparams.inference_inputs)
        infer_fn(hparams)
    elif mode == "train":
        train_fn(hparams)
    elif mode == "eval":
        eval_dir = hparams.eval_dir
        if not utils.check_file_existence(eval_dir):
            raise ValueError("Cannot find dir % s" % eval_dir)
        filenames = tf.gfile.ListDirectory(eval_dir)
        if len(filenames) == 0:
            raise ValueError("Can not find source under dir %s", eval_dir)
        for i in range(len(filenames)):
            filenames[i] = os.path.join(eval_dir, filenames[i])
            hparams.inference_inputs = filenames
        eval_fn(hparams)


def main():
    while True:
        hparams = standard_hparams_utils.Configuration()
        mode = input("Input execute mode['train' or 'infer' or 'eval']:")
        hparams.mode = mode
        if mode == "infer":
            infer_files = input("Input the image(s, split with ',') you want to generate caption(s):")
            infer_files_split = infer_files.split(",")
            beam_width = int(input("Input beam search width(default 1):"))
            if not beam_width:
                beam_width = 1
            hparams.inference_inputs = infer_files_split
            hparams.beam_width = beam_width
            config_msg = ''.join(["mode:", mode,
                                  ", infer_file:", infer_files,
                                  ", beam_width:", str(beam_width)])
        elif mode == "train":
            batch_size = int(input("Input batch size(default 32 for train&eval):"))
            train_steps = int(input("Set train step(default 400):"))
            if not train_steps:
                train_steps = 400
            if not batch_size:
                batch_size = 32
            hparams.batch_size = batch_size
            hparams.num_train_steps = train_steps
            config_msg = ''.join(["mode:", mode, ", batch_size:",
                                  str(batch_size), ", train_steps:",
                                  str(train_steps)])
        elif mode == "eval":
            """min_eval_steps = int(input("Set min evaluate step(default 500):"))
            if not min_eval_steps:
                min_eval_steps = 500
            eval_interval_secs = int(input("Set Interval seconds between two evaluation run(default=60):"))
            if not eval_interval_secs:
                eval_interval_secs = 60
            hparams.min_global_step = min_eval_steps
            hparams.eval_interval_secs = eval_interval_secs
            config_msg = ''.join(["mode:", mode, ", start_eval_steps:",
                                  str(min_eval_steps), ", interval_secs:",
                                  str(eval_interval_secs)])"""
            eval_dir = "D:/dataset/image_caption/val2014/"
            config_msg = ''.join(["eval_dir:", eval_dir])
            hparams.eval_dir = eval_dir
        else:
            raise ValueError("Unrecognized value %s " % mode)
        print(config_msg)
        train_fn = run_train.train
        infer_fn = run_inference.infer
        eval_fn = run_evaluate.evaluate
        run_main(hparams, train_fn, infer_fn, eval_fn)
        quit_sig = input("Quit or Continue?(Quit:q, Continue:c)")
        if quit_sig == "q":
            break
