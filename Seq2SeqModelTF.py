import tensorflow as tf

from BaseSeq2SeqModelTF import BaseSeq2Seq2ModelTF
from tensorflow.contrib import legacy_seq2seq

__author__ = "roopal_garg"

"""Sequence-to-sequence model with an attention mechanism."""


class Seq2SeqModelTF(BaseSeq2Seq2ModelTF):

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
        :param num_samples:
        :param fwd_only:
        """

        super(Seq2SeqModelTF, self).__init__(
            src_vocab_size, tgt_vocab_size, buckets, m, num_layers, mx_grad_nrm, batch_size, lr, model_name, save_dir,
            use_lstm=use_lstm, num_samples=num_samples, fwd_only=fwd_only
        )

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=2)
        self.train_writer.add_graph(graph=self.tf_session.graph, global_step=1)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(self, encoder_inputs, decoder_inputs, do_decode):
        return legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs, BaseSeq2Seq2ModelTF.get_cell_definition(self.M, self.num_layers),
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
        trainables = tf.trainable_variables()
        Seq2SeqModelTF.print_trainables(trainables)
        if not self.fwd_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.AdamOptimizer(self.lr)
            for b in xrange(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], trainables)

                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.mx_grad_nrm)

                self.gradient_norms.append(norm)

                self.updates.append(
                    opt.apply_gradients(zip(clipped_gradients, trainables), global_step=self.global_step)
                )
