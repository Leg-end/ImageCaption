"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import model_helper
from . import imc_model
from tensorflow.contrib import slim
import os

__all__ = ["get_model_creator",
           "train"]


def get_model_creator():
    """Get the right model class depending on configuration."""
    return imc_model.BaseModel


def train(hparams, scope=None):
    out_dir = hparams.out_dir
    print(out_dir)
    num_train_steps = hparams.num_train_steps

    # Create model
    model_creator = get_model_creator()  #
    train_model = model_helper.create_train_model(model_creator, hparams, scope)
    model = train_model.model
    print("create model")
    with train_model.graph.as_default():
        data_init_op = train_model.iterator.initializer
        pre_model_init_op = model.restore_CNN_op
        train_config = model.train_config()
        # Set up the learning rate.
        learning_rate_decay_fn = None
        if hparams.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (hparams.num_examples_per_epoch /
                                     hparams.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              hparams.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=hparams.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn
        train_op = tf.contrib.layers.optimize_loss(
            loss=train_config.train_loss,
            global_step=train_config.global_step,
            learning_rate=train_config.learning_rate,
            optimizer=train_config.optimizer,
            clip_gradients=train_config.grad_norm,
            learning_rate_decay_fn=learning_rate_decay_fn)
    saver = model.saver
    # Run training.
    print("Ready to train the model")
    slim.fully_connected()
    slim.learning.train(
        train_op,
        logdir=os.path.join(out_dir, "ckpt"),
        log_every_n_steps=hparams.steps_per_stats,
        graph=train_model.graph,
        global_step=train_config.global_step,
        number_of_steps=num_train_steps,
        local_init_op=data_init_op,
        init_fn=pre_model_init_op,
        saver=saver)


