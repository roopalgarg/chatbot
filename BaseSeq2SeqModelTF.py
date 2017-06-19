import tensorflow as tf
import numpy as np
import logging
import os
import time
import math
import random

import data_util

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, MultiRNNCell
from abc import ABCMeta, abstractmethod

__author__ = "roopal_garg"

TOKENS = data_util.Tokens


class BaseSeq2Seq2ModelTF:
    __metaclass__ = ABCMeta

    def __init__(
            self, src_vocab_size, tgt_vocab_size, buckets, m, num_layers, mx_grad_nrm, batch_size, lr, model_name,
            save_dir, use_lstm=False, num_samples=512, fwd_only=False
    ):
        """

        :param src_vocab_size:
        :param tgt_vocab_size:
        :param buckets:
        :param m:
        :param num_layers:
        :param mx_grad_nrm:
        :param batch_size:
        :param lr:
        :param use_lstm:
        :param num_samples:
        :param fwd_only:
        """
        tf.reset_default_graph()
        self.tf_session = tf.Session()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.buckets = buckets
        self.M = m
        self.num_layers = num_layers
        self.mx_grad_nrm = mx_grad_nrm
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.fwd_only = fwd_only

        # self.save_word_emb = save_word_emb
        self.model_name = "model_{model_name}".format(model_name=model_name)
        self.save_dir = os.path.join(save_dir, model_name)
        self.model_path = os.path.join(self.save_dir, "{}.ckpt".format(self.model_name))
        self.emb_path = os.path.join(self.save_dir, "word_embedding.npy")
        self.last_model_path = None

        self.lr = tf.Variable(float(lr), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        """ if we use sampled softmax, we need an output projection """
        self.output_projection = None
        self.softmax_loss_function = None

        """ sampled softmax only makes sense if we sample less than vocabulary size """
        if num_samples and num_samples < self.tgt_vocab_size:
            self.W = tf.get_variable("proj_w", [self.M, self.tgt_vocab_size])
            self.W_t = tf.transpose(self.W)

            self.b = tf.get_variable("proj_b", [self.tgt_vocab_size])
            self.output_projection = (self.W, self.b)

            self.softmax_loss_function = self.sampled_loss

        # if use_lstm:
        #     self.single_cell = BasicLSTMCell(self.M)
        # else:
        #     self.single_cell = GRUCell(self.M)

        # if self.num_layers > 1:
        #     self.cell = MultiRNNCell([GRUCell(self.M) for _ in range(self.num_layers)])

        self.cell = BaseSeq2Seq2ModelTF.get_cell_definition(self.M, self.num_layers)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i))
            )
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i))
            )
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        self.targets = [
            self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)
        ]

        self.losses = None
        self.gradient_norms = None
        self.updates = None
        self.outputs = None
        self.saver = None

    @staticmethod
    def get_cell_definition(m, num_layers):
        return MultiRNNCell([GRUCell(m) for _ in range(num_layers)])

    def add_summary_file_writer(self):
        logging.info("creating filewriter")
        return tf.summary.FileWriter(self.save_dir, graph=self.tf_session.graph)

    @abstractmethod
    def build_model(self):
        pass

    def sampled_loss(self, labels, inputs):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(
            self.W_t, self.b, labels, inputs, self.num_samples, self.tgt_vocab_size
        )

    @staticmethod
    def validate_inputs_with_bucket_size(encoder_size, decoder_size, enc_inp, dec_inp, tgt_wt):
        if len(enc_inp) != encoder_size:
            raise ValueError("encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(enc_inp), encoder_size))
        if len(dec_inp) != decoder_size:
            raise ValueError("decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(dec_inp), decoder_size))
        if len(tgt_wt) != decoder_size:
            raise ValueError("weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(tgt_wt), decoder_size))

    def predict(self, enc_inp, dec_inp, tgt_wt, bucket_id, fwd_only):
        """

        :param enc_inp:
        :param dec_inp:
        :param tgt_wt:
        :param bucket_id:
        :param fwd_only:
        :return:
        """
        encoder_size, decoder_size = self.buckets[bucket_id]

        self.validate_inputs_with_bucket_size(encoder_size, decoder_size, enc_inp, dec_inp, tgt_wt)

        """
        input feed: encoder inputs, decoder inputs, target_weights, as provided
        """
        input_feed_dict = dict()
        for l in xrange(encoder_size):
            input_feed_dict[self.encoder_inputs[l].name] = enc_inp[l]
        for l in xrange(decoder_size):
            input_feed_dict[self.decoder_inputs[l].name] = dec_inp[l]
            input_feed_dict[self.target_weights[l].name] = tgt_wt[l]

        """
        since our targets are decoder inputs shifted by one, we need one more.
        """
        last_target = self.decoder_inputs[decoder_size].name
        input_feed_dict[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        """
        output feed: depends on whether we do a backward step or not.
        """
        if not fwd_only:
            output_feed = [
                self.updates[bucket_id],  # Update Op that does SGD.
                self.gradient_norms[bucket_id],  # Gradient norm.
                self.losses[bucket_id]  # Loss for this batch.
            ]
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = self.tf_session.run(output_feed, input_feed_dict)

        if not fwd_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs

    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = list(), list()

        encoder_input = None
        decoder_input = None
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [TOKENS.PAD.idx] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append(
                [TOKENS.GO.idx] + decoder_input + [TOKENS.PAD.idx] * decoder_pad_size
            )

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = list(), list(), list()

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array(
                  [encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32
                )
            )

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array(
                    [decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32
                )
            )

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                target = None
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == TOKENS.PAD.idx:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def fit(
            self, enc_train, dec_train, enc_dev, dec_dev, max_train_data_size=None, test_every=500
    ):

        dev_set = data_util.read_data(enc_dev, dec_dev)
        train_set = data_util.read_data(enc_train, dec_train, max_size=max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(self.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        prev_loss = list()

        logging.info("initializing all variables")
        self.tf_session.run(tf.global_variables_initializer())

        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                train_set, bucket_id)

            _, step_loss, _ = self.predict(
                encoder_inputs, decoder_inputs, target_weights, bucket_id, False
            )

            step_time += (time.time() - start_time)
            loss += step_loss
            current_step += 1

            if current_step and current_step % test_every == 0:
                perplexity = math.exp(loss) if loss < test_every else float('inf')
                logging.info(
                    "global step {} step-time {} perplexity {}".format(
                        self.global_step.eval(self.tf_session), step_time, perplexity
                    )
                )

                prev_loss.append(loss)
                self.save_model(step=current_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(self.buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        logging.info("\teval: empty bucket {}: ".format(bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                        dev_set, bucket_id
                    )
                    _, eval_loss, _ = self.predict(
                        encoder_inputs, decoder_inputs, target_weights, bucket_id, True
                    )

                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    logging.info("\teval: bucket {} perplexity {}".format(bucket_id, eval_ppx))

    def close_session(self):
        self.tf_session.close()

    @staticmethod
    def print_trainables(trainables):
        for idx, trainable in enumerate(trainables):
            logging.info("Trainable: {} : {} : {}".format(idx+1, trainable.name, trainable))

    def get_latest_model_path(self):
        checkpoint_state = tf.train.get_checkpoint_state(self.save_dir)
        latest_model_path = checkpoint_state.model_checkpoint_path
        return latest_model_path

    def restore_latest_model(self):
        latest_model_path = self.get_latest_model_path()
        logging.info("loading model from {}".format(latest_model_path))
        self.saver.restore(self.tf_session, latest_model_path)

    def save_model(self, step=1):
        logging.info("saving model for step {} to {}".format(step, self.model_path))
        self.saver.save(self.tf_session, self.model_path, step)
        # if self.save_word_emb:
        #     self.save_embedding_matrix()

    # def save_embedding_matrix(self):
    #     logging.info("saving embedding matrix to {}".format(self.emb_path))
    #     np.save(self.emb_path, self.We.eval(self.tf_session))
