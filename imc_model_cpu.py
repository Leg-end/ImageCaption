"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from . import model_helper
from .utils import iterator_utils
from .utils import vocab_utils

__all__ = ["BaseModel"]


class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_loss", "optimizer",
                         "global_step",
                         "grad_norm", "learning_rate"))):
    pass


class EvalOutputTuple(collections.namedtuple("EvalOutputTuple",
                                             ("eval_loss", "predict_count", "batch_size"))):
    """To allow for flexibily in returning different outputs."""
    pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("infer_logits", "infer_summary", "sample_id",
                         "output_state", "sample_words", "scores"))):
    """To allow for flexibility in returning different outputs."""
    pass


class BaseModel(object):
    """Sequence-to-sequence base class.
    """

    def __init__(self,
                 hparams,
                 iterator,
                 reverse_vocab_table=None,
                 scope=None,
                 extra_args=None):

        self._set_params_initializer(hparams,
                                     iterator,
                                     scope, extra_args)
        # Train graph
        res = self.build_graph(scope=scope)
        self._set_train_or_infer(res, reverse_vocab_table)
        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), hparams.max_checkpoints_to_keep)

    def _setup_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP,
                         tf.GraphKeys.GLOBAL_VARIABLES])

    def _set_params_initializer(self,
                                hparams,
                                iterator,
                                extra_args=None,
                                scope=None):
        """Set various params for self and initialize."""
        self.hparams = hparams
        self.mode = hparams.mode
        self.dtype = tf.float32
        self.input_mask = None

        # Set data
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.batch_size = tf.constant(hparams.infer_batch_size)
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            resize_image = model_helper.process_image(image_feed,
                                                      self.is_training())
            self.images = tf.expand_dims(resize_image, 0)

            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")
            self.input_seqs = tf.expand_dims(input_feed, 1)
            self.word_id = tf.placeholder(dtype=tf.int64,
                                          shape=[], name="word_id")
            self.input_mask = None
            self.target_seqs = None
        else:
            assert isinstance(iterator, iterator_utils.BatchedInput)
            self.batch_size = tf.constant(hparams.batch_size)
            self.input_seqs = iterator.input_seqs
            self.target_seqs = iterator.target_seqs
            self.images = iterator.images
            self.input_mask = iterator.input_mask
            self.input_length = tf.reduce_sum(self.input_mask, axis=1)

        self.single_cell_fn = None
        if extra_args:
            self.single_cell_fn = extra_args.single_cell_fn

        # Global step
        self._setup_global_step()

        # Initializer
        self.initializer = model_helper.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.initializer_scale)
        # tf.get_variable_scope().set_initializer(self.initializer)

        # Embeddings
        if extra_args and extra_args.encoder_emb_lookup_fn:
            self.emb_lookup_fn = extra_args.encoder_emb_lookup_fn
        else:
            self.emb_lookup_fn = tf.nn.embedding_lookup

        # Other variables
        # cnn relative
        self.train_cnn = hparams.train_cnn
        self.CNN_variables = []
        self.restore_CNN_op = None
        # train relative
        self.target_cross_entropy_loss_weights = None
        self.target_cross_entropy_losses = None
        self.batch_loss = None

    def _set_train_or_infer(self, res, reverse_vocab_table):
        # word_count and predict_count not conclude
        mode = self.mode
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
        elif mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            self.reverse_vocab_table = reverse_vocab_table
            self.infer_logits, _, self.final_context_state = res
            # when do infer manually
            self.sample_words = reverse_vocab_table.lookup(
                    tf.to_int64(self.word_id))
        else:
            raise ValueError("Unknown mode type %s" % mode)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            # Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(
                self.input_length)

        params = tf.trainable_variables()

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.name_scope("Train_Config"):
                # compute ppl
                train_ppl = tf.exp(tf.div(self.train_loss, tf.to_float(self.predict_count)))
                tf.summary.scalar("train_ppl", train_ppl)

                if self.train_cnn:
                    learning_rate = self.hparams.cnn_learning_rate
                else:
                    learning_rate = self.hparams.initial_learning_rate
                self.learning_rate = tf.constant(learning_rate, dtype=self.dtype)
                tf.summary.scalar("losses/batch_loss", self.batch_loss)
                tf.summary.scalar("losses/total_loss", self.train_loss)
                # trainable variable summary
                for var in params:
                    tf.summary.histogram("parameters/" + var.op.name, var)
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary()

        print("# Trainable variables")
        print("Format: <name>, <shape>, <(soft) device placement>")
        for param in params:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

    def CNN_layer(self, scope=None):
        image_sequence = model_helper.get_CNN_output_for_image_caption(self.images,
                                                                       trainable=self.train_cnn,
                                                                       is_training=self.is_training())
        image_size = image_sequence.shape[1].value
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            # Restore CNN op
            cnn_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
            checkpoint_file = self.hparams.CNN_checkpoint_file
            saver = tf.train.Saver(cnn_variables)
            self.restore_CNN_op = model_helper.restore_CNN(checkpoint_file, saver)
        return image_sequence, image_size

    def image_embedding_layer(self,
                              cnn_image,
                              image_size,
                              scope=None):
        with tf.variable_scope(scope or "image_embedding", dtype=self.dtype) as img_scope:
            with tf.device("/cpu:0"):
                image_embedding = tf.contrib.layers.fully_connected(
                    inputs=cnn_image,
                    num_outputs=self.hparams.embedding_size,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    biases_initializer=None,
                    scope=img_scope)
            tf.constant(self.hparams.embedding_size, name="embedding_size")
        return image_embedding

    def input_embedding_layer(self,
                              scope=None,
                              embed_name="input_embedding_layer"):
        with tf.variable_scope(scope or "input_embedding", dtype=self.dtype):
            # better to put it on cpu
            embedding_layer = model_helper.create_or_load_embed(
                embed_name=embed_name,
                vocab_file=self.hparams.tgt_vocab_file,
                embed_file=self.hparams.tgt_embed_file,
                vocab_size=self.hparams.cap_vocab_size,
                device_str="/cpu:0",
                embed_size=self.hparams.embedding_size,
                initializer=self.initializer)
            input_embedding = self.emb_lookup_fn(embedding_layer, self.input_seqs)
        return input_embedding

    def projection_layer(self, outputs, scope=None):
        with tf.variable_scope("projection_layer") as logits_scope:
            """num_layers = self.hparams.num_layers
            num_gpus = self.hparams.num_gpus
            device_id = num_layers if num_layers < num_gpus else (
                    num_layers - 1)"""
            with tf.device("/cpu:0"):
                logits = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=self.hparams.cap_vocab_size,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    scope=logits_scope)
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                logits = tf.nn.softmax(logits, name="softmax")
        return logits

    def _build_cell(self, base_gpu=0):
        cell = model_helper.create_rnn_cell(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=self.hparams.num_layers,
            forget_bias=self.hparams.forget_bias,
            dropout=self.hparams.dropout,
            mode=self.mode,
            num_gpus=self.hparams.num_gpus,
            state_is_tuple=True,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)
        return cell

    def build_cell_state(self, cell, image_embedding):
        # Encoder_outputs: [max_time, batch_size, num_units]
        zero_state = cell.zero_state(
            batch_size=self.batch_size, dtype=self.dtype)
        cell_outputs, cell_state = cell(image_embedding, zero_state)
        return cell_outputs, cell_state

    def infer_manually(self, cell, cell_initial_state, input_embedding):
        # pass out for inference
        tf.concat(axis=1, values=cell_initial_state, name="cell_initial_state")
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
        outputs, state_tuple = cell(
            inputs=tf.squeeze(input_embedding, axis=[1]),
            state=state_tuple)
        tf.concat(axis=1, values=state_tuple, name="state")
        return outputs, state_tuple
    # infer by seq2seq
    """def infer_sequentially(self,
                           infer_mode,
                           cell,
                           cell_scope,
                           cell_initial_state,
                           projection_layer):
        logits = tf.no_op()
        my_decoder = None
        tos_id = vocab_utils.TOS_ID
        toe_id = vocab_utils.TOE_ID
        start_tokens = tf.fill([self.hparams.infer_batch_size], tos_id)
        end_token = toe_id
        helper = None
        if infer_mode == "beam_search":
            beam_width = self.hparams.beam_width
            length_penalty_weight = self.hparams.length_penalty_weight
            coverage_penalty_weight = self.hparams.coverage_penalty_weight
            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=self.inp_embedding_layer,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=cell_initial_state,
                beam_width=beam_width,
                output_layer=projection_layer,
                length_penalty_weight=length_penalty_weight,
                coverage_penalty_weight=coverage_penalty_weight)
        elif infer_mode == "greedy":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.inp_embedding_layer, start_tokens, end_token)
        else:
            raise ValueError("Unknown infer_mode '%s'", infer_mode)
        if infer_mode != "beam_search":
            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                helper,
                cell_initial_state,
                output_layer=self.pro_layer  # applied per timestep
            )
            # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=self.hparams.max_infer_iterations,
            output_time_major=self.hparams.time_major,
            swap_memory=True,
            scope=cell_scope)
        if infer_mode == "beam_search":
            sample_id = outputs.predicted_ids
            scores = outputs.scores
        else:
            # compute logprob of the max one
            logits = tf.log(tf.nn.softmax(outputs.rnn_output), name="logprop")
            max_index = tf.arg_max(logits)
            sample_id = outputs.sample_id
            scores = logits[max_index]
        return logits, outputs, final_context_state, sample_id, scores"""

    def dynamic_rnn_layer(self,
                          image_embedding,
                          input_embedding,
                          base_gpu=0, scope=None):
        cell = self._build_cell(base_gpu)
        with tf.variable_scope(scope or "dynamic_rnn_layer", initializer=self.initializer) as cell_scope:
            # build initial state from image embedding
            _, cell_initial_state = self.build_cell_state(cell, image_embedding)
            # Allow the LSTM variables to be reused.
            # reuse the scope where we build cell's variable, then wen can call the same cell repeatedly
            cell_scope.reuse_variables()

            # Train or eval-------------------------------------------------------------------------------
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                outputs, state_tuple = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=input_embedding,
                    sequence_length=self.input_length,
                    initial_state=cell_initial_state,
                    dtype=self.dtype,
                    scope=cell_scope)

            # Inference----------------------------------------------------------------------------------
            else:
                outputs, state_tuple = self.infer_manually(cell,
                                                           cell_initial_state,
                                                           input_embedding)
            outputs = tf.reshape(outputs, [-1, cell.output_size])
            return outputs, state_tuple

    def loss_func(self, logits, outputs, scope=None):
        with tf.name_scope(scope or "loss"):
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                """with tf.device(model_helper.get_device_str(
                        self.hparams.num_layers - 1,
                        self.hparams.num_gpus)):"""
                with tf.device("/cpu:0"):  # when we compute loss in gpu, will cause NAN global norm
                    batch_loss, total_loss, crossent, mask_weights = self._compute_loss(logits, outputs)
                self.batch_loss = batch_loss
                self.target_cross_entropy_losses = crossent  # Used in evaluation.
                self.target_cross_entropy_loss_weights = mask_weights  # Used in evaluation.
            else:
                total_loss = tf.constant(0.0)
                self.batch_loss = tf.constant(0.0)
        return total_loss

    def _compute_loss(self, logits, cell_outputs):
        """Compute optimization loss.
        Args:

        Returns:

        Raises:
        """
        target_seqs = self.target_seqs
        target_weights = self.input_mask
        if self.hparams.time_major:
            target_seqs = tf.transpose(target_seqs)
        target_seqs = tf.reshape(target_seqs, [-1])
        crossent = self._softmax_cross_entropy_loss(
            logits, cell_outputs, target_seqs)
        # target_weight alias input_mask, to get rid of redundance part of words,
        # e.g.<tos >/<toe>filled region
        if self.hparams.time_major:
            target_weights = tf.transpose(target_weights)
        target_weights = tf.to_float(tf.reshape(target_weights, [-1]))
        batch_loss = tf.div(tf.reduce_sum(tf.multiply(crossent, target_weights)),
                            tf.reduce_sum(target_weights), name="batch_loss")
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()
        return batch_loss, total_loss, crossent, target_weights

    def _softmax_cross_entropy_loss(
            self, logits, decoder_cell_outputs, labels):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        return crossent

    def build_graph(self, scope=None):
        """SubClass must implement this method.
        Creates a sequence-to-sequence model with dynamic RNN decoder API.
        Args:

        Returns:
            logits:
            loss:
            final_context_state:
            sample_id:

        Raises:
        """
        # CNN
        cnn_image, image_size = self.CNN_layer()

        # image embedding
        image_embedding = self.image_embedding_layer(cnn_image, image_size)

        # caption embedding
        input_embedding = self.input_embedding_layer()

        # build network
        final_output, final_state_tuple = self.dynamic_rnn_layer(image_embedding,
                                                                 input_embedding)

        # projection layer
        logits = self.projection_layer(final_output)

        # loss
        loss = self.loss_func(logits, final_output)

        return logits, loss, final_state_tuple

    def get_max_time(self, tensor):
        """Get sequence length"""
        time_axis = 0 if self.hparams.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def is_training(self):
        return self.mode == tf.contrib.learn.ModeKeys.TRAIN

    # invoked by outside executor ----------------------------------------------------------------------------
    def train_config(self):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        output_tuple = TrainOutputTuple(train_loss=self.train_loss,
                                        optimizer=self.hparams.optimizer,
                                        global_step=self.global_step,
                                        grad_norm=self.hparams.max_gradient_norm,
                                        learning_rate=self.learning_rate)
        return output_tuple

    def eval(self, sess):
        """Execute eval graph."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        output_tuple = EvalOutputTuple(eval_loss=self.eval_loss,
                                       predict_count=self.predict_count,
                                       batch_size=self.batch_size)
        return sess.run(output_tuple)

    def _get_infer_summary(self):
        return tf.no_op()

    # -----------------------------------------do infer manually---------------------------------
    def eval_initial_state(self, sess, image_feed):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        initial_state = sess.run(fetches=
                                 "dynamic_rnn_layer/cell_initial_state:0",
                                 feed_dict={"image_feed:0": image_feed})
        return initial_state

    def infer_step(self, sess, input_feed, state_tuple_feed):
        softmax, state = sess.run(
            fetches=["projection_layer/softmax:0", "dynamic_rnn_layer/state:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "dynamic_rnn_layer/state_feed:0": state_tuple_feed,
            })
        return softmax, state, None

    def decode(self, sess, word_id):
        return sess.run(self.sample_words, feed_dict={"word_id:0": word_id})

    # -----------------------------------------do infer sequentially-------------------------------
    def infer_config(self, sess, image_feed):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        output_tuple = InferOutputTuple(infer_logits=self.infer_logits,
                                        infer_summary=self.infer_summary,
                                        sample_id=self.sample_id,
                                        sample_words=self.sample_words,
                                        scores=self.scores)
        return sess.run(output_tuple, feed_dict={"image_feed:0": image_feed})

    def decode_sequentially(self, sess, image_feed):
        output_tuple = self.infer_config(sess, image_feed)
        sample_words = output_tuple.sample_words
        scores = output_tuple.scores
        infer_summary = output_tuple.infer_summary
        # make sure outputs is of shape [batch_size, time] or [beam_width,
        # batch_size, time] when using beam search.
        if self.hparams.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3:
            # beam search output in [batch_size, time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])
        return sample_words, scores, infer_summary
