import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, MultiRNNCell
from tensorflow.contrib import legacy_seq2seq
"""Sequence-to-sequence model with an attention mechanism."""


class Seq2SeqModelTF:

    def __init__(
            self, src_vocab_size, tgt_vocab_size, buckets, m, num_layers, mx_grad_nrm, batch_size, lr, use_lstm=False,
            num_samples=512, fwd_only=False
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
        :param num_samples:
        :param fwd_only:
        """

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.buckets = buckets
        self.M = m
        self.num_layers = num_layers
        self.mx_grad_nrm = mx_grad_nrm
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.fwd_only = fwd_only

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

        if use_lstm:
            self.single_cell = BasicLSTMCell(self.M)
        else:
            self.single_cell = GRUCell(self.M)

        if self.num_layers > 1:
            self.cell = MultiRNNCell([self.single_cell] * self.num_layers)

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

        self.build_model()

    def sampled_loss(self, inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(
            self.W_t, self.b, inputs, labels, self.num_samples, self.tgt_vocab_size
        )

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(self, encoder_inputs, decoder_inputs, do_decode):
        return legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs, self.cell,
            num_encoder_symbols=self.src_vocab_size,
            num_decoder_symbols=self.tgt_vocab_size,
            embedding_size=self.M,
            output_projection=self.output_projection,
            feed_previous=do_decode
        )

    def build_model(self):
        # Training outputs and losses.
        if self.fwd_only:
            self.outputs, self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.target_weights, self.buckets, lambda x, y: self.seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function
            )

            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection is not None:
                for b in xrange(len(self.buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.target_weights, self.buckets,
                lambda x, y: self.seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function
            )

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not self.fwd_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.AdamOptimizer(self.lr)
            for b in xrange(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params)

                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.mx_grad_nrm)

                self.gradient_norms.append(norm)

                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())
