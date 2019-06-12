"""For loading data into ImageCaption models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import image_process_utils
from . import vocab_utils

import collections
import tensorflow as tf

__all__ = ["get_infer_iterator", "get_iterator"]


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "images", "input_seqs",
                            "target_seqs", "input_mask"))):
    pass


def get_iterator(src_dataset,
                 batch_size,
                 image_resize_height=229,
                 image_resize_width=229,
                 image_format="jpeg",
                 image_channels=3,
                 num_parallel_calls=4,
                 output_buffer_size=10,
                 image_feature="image/data",
                 caption_feature="image/caption_ids"):

    def get_feature_description(image_feature_, caption_feature_):
        context_features_proto = {
            image_feature_: tf.FixedLenFeature([], dtype=tf.string),
        }
        sequence_features_proto = {
            caption_feature_: tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        return context_features_proto, sequence_features_proto
    
    context_features, sequence_features = get_feature_description(
        image_feature, caption_feature)

    # fetch context and sequence from record
    src_dataset = src_dataset.map(
        lambda x: (tf.parse_single_sequence_example(
            x, context_features, sequence_features)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # fetch image data from context and process it, fetch caption from sequence
    src_dataset = src_dataset.map(
        lambda context, sequence: (
            image_process_utils.process_image(
                context[image_feature],
                is_training=True,
                height=image_resize_height,
                width=image_resize_width,
                thread_id=0,
                image_format=image_format),
            # image_process_utils.process_image returns (image, image_summarys) in mode train, image_summary is None
            tf.to_int32(sequence[caption_feature])),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # create a caption-in prefixed with <tos> and a caption-out suffixed with <toe>.
    src_dataset = src_dataset.map(
        lambda img, cap: (img,
                          cap[:-1],
                          cap[1:]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # add input mask,for cap_in=["1","2","3","1","2","3"], input_mask is [1, 1, 1, 1, 1, 1]
    src_dataset = src_dataset.map(
        lambda img, cap_in, cap_out: (
            img, cap_in, cap_out, tf.ones(shape=[tf.size(cap_in)], dtype=tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([image_resize_width, image_resize_height, image_channels]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None])),
            padding_values=(
                float(0),
                vocab_utils.TOE_ID,
                vocab_utils.TOE_ID,
                0),
            drop_remainder=True)
    # padded batch
    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_image, caption_in, caption_out, caption_mask) = batched_iter.get_next()
    lengths = tf.add(tf.reduce_sum(caption_mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

    return BatchedInput(
        initializer=batched_iter.initializer,
        images=src_image,
        input_seqs=caption_in,
        target_seqs=caption_out,
        input_mask=caption_mask)


def get_infer_iterator(src_dataset,
                       batch_size,
                       image_resize_height=229,
                       image_resize_width=229,
                       image_format="jpeg",
                       image_channels=3):
    # fetch image data from context and process it, fetch caption from sequence
    src_dataset = src_dataset.map(
        lambda img:
            (image_process_utils.process_image(
                img,
                is_training=False,
                height=image_resize_height,
                width=image_resize_width,
                thread_id=0,
                image_format=image_format)))
    
    # batch
    batched_dataset = src_dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_image = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        images=src_image,
        input_seqs=None,
        target_seqs=None,
        input_mask=None)

