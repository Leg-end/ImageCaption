"""Helper functions for image preprocessing"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

__all__ = ["distort_image", "process_image"]


def distort_image(image, thread_id):
    """Perform random distortions on an image.

    Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
    distortions. There should be a multiple of 2 preprocessing threads.

    Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
    [0, 1].
    """

    # Randomly flip horizontally.
    with tf.name_scope("flip_horizontal", values=[image]):
        image = tf.image.random_flip_left_right(image)
    # Randomly distort the colors based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format="jpeg"):

    def image_summary(name, image_):
        if not thread_id:
            tf.summary.image(name, tf.expand_dims(image_, 0))

    with tf.name_scope("decode", values=[encoded_image]):
        
        if image_format == "jpeg":
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_summary("original_image", image)

    assert(resize_height > 0) == (resize_width > 0)
    if resize_height:
        image = tf.image.resize_images(image,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)

    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    image_summary("resized_image", image)
    # Randomly distort the image.

    if is_training:
        image = distort_image(image, thread_id)

    image_summary("final_image", image)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
