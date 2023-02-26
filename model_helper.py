"""Utility functions for building models"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import numpy as np
import six
import tensorflow as tf

from .utils import iterator_utils
from .utils import vocab_utils
from .utils import misc_utils as utils
from .utils import image_process_utils
from . import image_embedding 


VOCAB_SIZE_THRESHOLD_CPU = 50000

__all__ = ["get_initializer", "get_device_str",
           "create_train_model", "create_eval_model",
           "create_infer_model", "get_embed_device",
           "get_CNN_output_for_image_caption", "restore_CNN",
           "_create_pretrained_emb_from_txt", "create_or_load_embed",
           "create_image_embedding_encoder", "create_emb_for_encoder_and_decoder",
           "_single_cell", "_cell_list", "create_rnn_cell",
           "gradient_clip", "print_variables_in_ckpt",
           "load_model", "create_or_load_model"]


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)
    

def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


class ExtraArgs(collections.namedtuple(
    "ExtraArgs", ("single_cell_fn", "model_device_fn",
                  "attention_mechanism_fn", "encoder_emb_lookup_fn"))):
    pass


class TrainModel(collections.namedtuple("TrainModel",
                                        ("graph", "model", "iterator"))):
    
    pass


def create_train_model(model_creator,
                       hparams,
                       scope=None,
                       extra_args=None):
    dataset_dir = os.path.join(hparams.base_path, "train")
    print("dataset_dir: %s" % dataset_dir)
    tgt_vocab_file = hparams.tgt_vocab_file
    if not utils.check_file_existence(dataset_dir):
        raise ValueError("Dataset dir %s not found.", dataset_dir)
    filenames = tf.gfile.ListDirectory(dataset_dir)
    if len(filenames) == 0:
        raise ValueError("Can not find source under dir %s", dataset_dir)
    if not tgt_vocab_file and not utils.check_file_existence(tgt_vocab_file):
        raise ValueError("Vocabulary file %s not found.", hparams.tgt_vocab_file)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(dataset_dir, filenames[i])
    print("train filenames:", filenames)
    graph = tf.Graph()
    
    with graph.as_default(), tf.container(scope or "train"):
        with tf.name_scope("Data"):
            raw_dataset = tf.data.TFRecordDataset(filenames)

            iterator = iterator_utils.get_iterator(
                raw_dataset,
                hparams.batch_size,
                image_resize_height=hparams.image_height,
                image_resize_width=hparams.image_width,
                image_format=hparams.image_format,
                image_channels=hparams.image_channels,
                num_parallel_calls=hparams.num_parallel_calls,
                output_buffer_size=hparams.output_buffer_size,
                image_feature=hparams.image_feature_name,
                caption_feature=hparams.caption_feature_name)

        # model_device_fn = None
        # if extra_args:
            # model_device_fn = extra_args.model_device_fn
        # with tf.device(model_device_fn):# None ?
        model = model_creator(
            hparams=hparams,
            iterator=iterator,
            scope=scope,
            extra_args=extra_args)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_files_placeholder",
                            "iterator"))):
    pass


def create_eval_model(model_creator,
                      hparams,
                      scope=None,
                      extra_args=None):
    graph = tf.Graph()
    vocab_file = hparams.tgt_vocab_file
    if not vocab_file or not utils.check_file_existence(vocab_file):
        raise ValueError("Vocabulary file %s not found", vocab_file)

    with graph.as_default(), tf.container(scope or "eval"):
        with tf.name_scope("Data"):
            reverse_vocab_table = vocab_utils.create_reverse_vocab_table(vocab_file)
            dataset_files_placeholder = tf.placeholder(shape=[None], dtype=tf.string) # filename list
            raw_dataset = tf.data.TFRecordDataset(dataset_files_placeholder)

            iterator = iterator_utils.get_iterator(
                raw_dataset,
                hparams.batch_size,
                image_resize_height=hparams.image_height,
                image_resize_width=hparams.image_width,
                image_format=hparams.image_format,
                image_channels=hparams.image_channels,
                image_feature=hparams.image_feature_name,
                caption_feature=hparams.caption_feature_name)
        model = model_creator(
            hparams,
            iterator=iterator,
            reverse_vocab_table=reverse_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return EvalModel(
        graph=graph,
        model=model,
        src_files_placeholder=dataset_files_placeholder,
        iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
    pass


def create_infer_model(model_creator,
                       hparams,
                       scope=None,
                       extra_args=None):
    graph = tf.Graph()
    vocab_file = hparams.tgt_vocab_file
    if not vocab_file or not utils.check_file_existence(vocab_file):
        raise ValueError("Vocabulary file %s not found", vocab_file)

    with graph.as_default(), tf.container(scope or "infer"):
        with tf.name_scope("Data"):
            reverse_vocab_table = vocab_utils.create_reverse_vocab_table(vocab_file)
            # In infer mode, we use placeholder instead of dataset
        model = model_creator(
            hparams,
            iterator=None,
            reverse_vocab_table=reverse_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return InferModel(
        graph=graph,
        model=model)


def process_image(image,
                  is_training,
                  image_height=299,
                  image_width=299,
                  thread_id=0,
                  image_format="jpeg"):
    return image_process_utils.process_image(image,
                                             is_training=is_training,
                                             height=image_height,
                                             width=image_width,
                                             thread_id=thread_id,
                                             image_format=image_format)


def get_embed_device(vocab_size):#?
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


# CNN model relative
#################################################################
def get_CNN_output_for_image_caption(img,
                                     trainable=False,
                                     is_training=False):
    return image_embedding.inception_v3(
            img,
            trainable=trainable,
            is_training=is_training)


def restore_CNN(ckpt_file, saver):
    restore_fn = tf.no_op()
    if ckpt_file and utils.check_file_existence(ckpt_file):
        def restore_fn(sess):
            tf.logging.info("Restoring CNN variables from checkpoint file %s",
                            ckpt_file)
            saver.restore(sess, ckpt_file)
    return restore_fn
########################################################################


def _create_pretrained_emb_from_txt(vocab_file,
                                    embed_file,
                                    num_trainable_tokens=3,
                                    dtype=tf.float32,
                                    scope=None):
    print("debug")
    vocab, _ = vocab_utils.load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]  # adding unk, start and stop are trainable

    print("# Using pretrained embedding: %s." % embed_file)
    print("  with trainable tokens: ")

    emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
    for token in trainable_tokens:
        utils.print_out("    %s" % token)
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size

    emb_mat = np.array(
        [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    with tf.variable_scope(scope or "pretrained_embeddings", dtype=dtype) as scope:
        with tf.device(get_embed_device(num_trainable_tokens)):
            emb_mat_var = tf.get_variable(
                "emb_mat_var", [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0)  # concat pretrained embbed vectors with 3 new trainable vectors


def create_or_load_embed(embed_name, vocab_file,
                         embed_file, vocab_size,
                         embed_size, device_str=None,
                         initializer=None, dtype=None):
    if embed_file and vocab_file:
        print("embed_file:", embed_file)
        embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
    else:
        with tf.device(device_str or get_embed_device(vocab_size)):
            embedding = tf.get_variable(
                embed_name, shape=[vocab_size, embed_size],
                initializer=initializer, dtype=dtype)
    return embedding


def create_image_embedding_encoder(CNN_output_size,
                                   embed_size,
                                   weights_initializer,
                                   name="image_embedding_layer"):
    with tf.device(get_embed_device(CNN_output_size)):
        image_embedding_layer = tf.layers.Dense(
                units=embed_size,
                activation=None,
                use_bias=False,
                kernel_initializer=weights_initializer,
                bias_initializer=None,
                name=name)
    return image_embedding_layer


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_enc_partitions=0,
                                       num_dec_partitions=0,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None,
                                       src_embed_file=None,
                                       tgt_embed_file=None,
                                       scope=None):
    
    if num_enc_partitions <= 1:
        enc_partitioner = None
    else:
        enc_partitioner = tf.fixed_size_partitioner(num_enc_partitions)

    if num_dec_partitions <= 1:
        dec_partitioner = None
    else:
        dec_partitioner = tf.fixed_size_partitioner(num_dec_partitions)

    if src_embed_file and enc_partitioner:
        raise ValueError(
            "Can't set num_enc_partitions > 1 when using pretrained encoder embedding")
    if tgt_embed_file and enc_partitioner:
        raise ValueError(
            "Can't set num_dec_partitions > 1 when using pretrained decoder embedding")

    with tf.variable_scope(
            scope or "embeddings", dtype=dtype, partitioner=enc_partitioner) as scope:
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab sizes"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            assert src_embed_size == tgt_embed_size
            utils.print_out("# Use the same embedding for source and target")
            vocab_file = src_vocab_file or tgt_vocab_file
            embed_file = src_embed_file or tgt_embed_file

            embedding_encoder = create_or_load_embed(
                "embedding_share", vocab_file, embed_file,
                src_vocab_size, src_embed_size, dtype)
            embedding_decoder = embedding_encoder
        else:
            with tf.variable_scope("encoder", partitioner=enc_partitioner):
                embedding_encoder = create_or_load_embed(
                    "embedding_encoder", src_vocab_file, src_embed_file,
                    src_vocab_size, src_embed_size, dtype)

            with tf.variable_scope("decoder", partitioner=dec_partitioner):
                embedding_decoder = create_or_load_embed(
                    "embedding_decoder", tgt_vocab_file, tgt_embed_file,
                    tgt_vocab_size, tgt_embed_size, dtype)

    return embedding_encoder, embedding_decoder


def _single_cell(unit_type,
                 num_units,
                 dropout,
                 mode,
                 forget_bias=1.0,
                 state_is_tuple=False,
                 device_str=None):
    """Create an instance of single RNN cell"""
    # dropout = (1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    # Cell Type
    if unit_type == "lstm":
        print("    LSTM, forget_bias=%g" % forget_bias)
        single_cell = tf.nn.rnn_cell.LSTMCell(
            num_units,
            forget_bias=forget_bias,
            state_is_tuple=state_is_tuple)
    elif unit_type == "gru":
        print("    GRU")
        single_cell = tf.nn.rnn_cell.GRUCell(
            num_units,
            state_is_tuple=state_is_tuple)
    elif unit_type == "layer_norm_lstm":
        print("    Layer Normalized LSTM, forget_bias=%g" % forget_bias)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True,
            state_is_tuple=state_is_tuple)
    elif unit_type == "nas":
        utils.print_out("    NASCell", new_line=False)
        single_cell = tf.contrib.rnn.NASCell(num_units, state_is_tuple=state_is_tuple)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    if dropout > 0.0 and mode == tf.contrib.learn.ModeKeys.TRAIN:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=dropout, output_keep_prob=dropout)
        print("    %s, dropout_keep=%g " % (type(single_cell).__name__, dropout))

    # Device Wrapper
    if device_str:
        single_cell = tf.nn.rnn_cell.DeviceWrapper(single_cell, device_str)
        print("    %s, device=%s" %
              (type(single_cell).__name__, device_str))

    return single_cell


def _cell_list(unit_type,
               num_units,
               num_layers,
               forget_bias,
               dropout,
               mode,
               num_gpus,
               base_gpu=0,
               state_is_tuple=False,
               single_cell_fn=None):
    " Create a list of RNN cells "
    if not single_cell_fn:
        single_cell_fn = _single_cell

    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        print("    cell %d" % i)
        single_cell = single_cell_fn(unit_type=unit_type,
                                     num_units=num_units,
                                     forget_bias=forget_bias,
                                     dropout=dropout,
                                     mode=mode,
                                     state_is_tuple=state_is_tuple,
                                     device_str=get_device_str(i + base_gpu, num_gpus))
        print("")
        cell_list.append(single_cell)
    return cell_list


def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    forget_bias,
                    dropout,
                    mode,
                    num_gpus,
                    state_is_tuple=False,
                    base_gpu=0,
                    single_cell_fn=None):
    """Create multi-layer RNN cell.
        Args:

        Returns:

        Raises:
        """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           state_is_tuple=state_is_tuple,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn)
    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:
        # Multi layers
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summary, gradient_norm


def print_variables_in_ckpt(ckpt_path):
    """Print a list of variables in a checkpoint together with their shapes."""
    print("# Variables in ckpt %s" % ckpt_path)
    reader = tf.train.NewCheckpointReader(ckpt_path)
    variables = []
    variables.append(reader.get_tensor("image_embedding/image_embedding_encoder/kernel"))
    variables.append(reader.get_tensor("encoder/encoder_cell/lstm_cell/kernel"))
    # variables.append(reader.get_tensor("encoder/encoder_cell/lstm_cell/bias"))
    # variables.append(reader.get_tensor("caption_embedding/caption_embedding_layer"))
    # variables.append(reader.get_tensor("output_projection/kernel"))
    for key in variables:
        print(key)


def load_model(model, ckpt_path, session, init_op=None, name="restore"):
    start_time = time.time()
    ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(ckpt_path))
    if ckpt and ckpt.model_checkpoint_path:
        if init_op:
            print("Restore a pretrained model")
            init_op(session)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        # print_variables_in_ckpt(ckpt.model_checkpoint_path)
    else:
        print("Can't load checkpoint")
        
    restore_op = tf.tables_initializer()
    # check if it returns no_op
    print(restore_op)
    session.run(restore_op)
    print(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt_path, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, init_op=None, name="restore"):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, init_op, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" %
              (name, time.time() - start_time))
    global_step = model.global_step.eval(session=session)
    print(global_step)
    return model, global_step

