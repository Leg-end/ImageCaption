"""standard hparams utils."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import vocab_utils


import tensorflow as tf

"""def create_standard_hparams():
    return tf.contrib.training.HParams(
        #Data
        base_path = "D:/image_caption/dataset/",
        src_vocab_file = None,
        tgt_vocab_file = "D:/image_caption/dataset/word_counts.txt",
        src_embed_file = None,
        tgt_embed_file = None,
        image_feature_name = "image/data",
        caption_feature_name = "image/caption",
        CNN_checkpoint_file = "D:/dataset/model/inception_v3.ckpt",
        out_dir = "D:/image_caption/ckpt/",

        #Data Constraint
        input_file_pattern = None,
        batch_size = 32,
        image_height = 299,
        image_width = 299,
        image_format = "jpeg",
        image_channels = 3,
        num_process_threads = 4,

        #Network
        initializer_scale = 0.08,
        embedding_size = 512,
        num_units = 512,
        dropout = 0.7,
        num_encoder_layers = 1,
        num_decoder_layers = 1,
        pass_hidden_state = True,
        encoder_type = "undi",
        unit_type = "lstm",
        forget_bias = 1.0,

        #Train
        num_sampled_softmax = 0,
        optimizer = "sgd",
        initial_learning_rate = 2.0,
        decay_scheme = "luong5",
        num_train_step = 50,
        warmup_steps = 0,
        warmup_scheme = "t2t",
        train_inception_learning_rate = 0.0005,
        max_gradient_norm = 5.0,
        max_checkpoints_to_keep = 5,
        is_CNN_trainable = False,
        colocate_gradients_with_ops = False,
        num_enc_emb_partitions = 0,
        num_dec_emb_partitions = 0,
        epoch_step = 0,
        
        
        #Inference
        infer_mode = "greedy",
        beam_width = 0,
        length_penalty_weight = 0.0,
        coverage_penalty_weight = 0.0,
        sampling_temperature = 0.0,
        

        #Misic
        log_device_placement = True,
        mode = "train",
        num_parallel_calls = 4,
        output_buffer_size = batch_size*1000,
        init_op = "uniform",
        random_seed = 12345,
        init_weight = None,
        num_gpus = 0,
        time_major = False,
        num_intra_threads = 0,
        num_inter_threads = 0,

        #Vocab
        cap_vocab_size = 0,
        tos = "<S>",
        toe = "</S>"
        )"""


class Configuration(object):
    def __init__(self):

        # Data
        self.base_path = "D:/image_caption/dataset/"
        self.src_vocab_file = None
        self.tgt_vocab_file = "D:/image_caption/dataset/word_vocab.txt"
        self.src_embed_file = None
        self.tgt_embed_file = None
        self.image_feature_name = "image/data"
        self.caption_feature_name = "image/caption_ids"
        self.CNN_checkpoint_file = "D:/dataset/model/inception_v3.ckpt"
        self.out_dir = "D:/image_caption/train_results/"
        self.inference_inputs = ["D:/dataset/image_caption/val2014/COCO_val2014_000000581593.jpg"]
        self.inference_output_dir = "D:/image_caption/infer_results/"

        # Data Constraint
        self.batch_size = 32
        self.image_height = 299
        self.image_width = 299
        self.image_format = "jpeg"
        self.image_channels = 3
        self.num_process_threads = 4

        # Network
        self.initializer_scale = 0.08
        self.embedding_size = 512
        self.num_units = 512
        self.dropout = 0.7
        self.num_layers = 1
        self.unit_type = "lstm"
        self.forget_bias = 1.0

        # Train
        self.num_sampled_softmax = 0
        self.optimizer = "SGD"
        self.initial_learning_rate = 2.0
        self.num_train_steps = 1000
        self.max_gradient_norm = 5.0
        self.max_checkpoints_to_keep = 5
        self.train_cnn = False
        self.cnn_learning_rate = 0.0005
        self.num_enc_emb_partitions = 0
        self.num_dec_emb_partitions = 0
        self.epoch_step = 0
        self.steps_per_stats = 5
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0
        self.num_examples_per_epoch = 586363
        
        # Inference
        self.infer_mode = "greedy"
        self.beam_width = 3
        self.length_penalty_weight = 0.0
        self.coverage_penalty_weight = 0.0
        self.sampling_temperature = 0.0
        self.inference_indices = []
        self.infer_batch_size = 1
        self.max_caption_length = 20

        # Evaluate
        self.min_global_step = 500
        self.eval_interval_secs = 60
        self.eval_dir = "D:/image_caption/eval_results/"
        self.num_eval_examples = 10132
        self.eval_output_dir = "D:/image_caption/eval_results"

        # Misic
        self.log_device_placement = True
        self.mode = "infer"
        self.num_parallel_calls = 4
        self.output_buffer_size = self.batch_size*10
        self.init_op = "uniform"
        self.random_seed = 12345
        self.num_gpus = 1
        self.time_major = False
        self.num_intra_threads = 0
        self.num_inter_threads = 0

        _, vocab_size = vocab_utils.load_vocab(self.tgt_vocab_file)
        # Vocab
        self.cap_vocab_size = vocab_size+1
        self.tos = "<S>"
        self.toe = "</S>"
