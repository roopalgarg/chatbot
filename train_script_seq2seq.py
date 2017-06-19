import sys
import logging

import data_util

from Seq2SeqModelTF import Seq2SeqModelTF

from config.ConfigHandler import ConfigHandler

__author__ = "roopal_garg"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s : %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

BUCKETS = data_util.BUCKETS

EXIT_PHRASE = ConfigHandler.get("exit_term", "model_param")


def main():
    logging.info("preparing data")
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_util.prepare_datasets()

    logging.info("initializing the model")

    lr = ConfigHandler.getfloat("learning_rate", "model_param")
    vocab_size_enc = ConfigHandler.getint("vocab_size_enc", "model_param")
    vocab_size_dec = ConfigHandler.getint("vocab_size_dec", "model_param")
    num_layers = ConfigHandler.getint("num_layers", "model_param")
    mx_grad_nrm = ConfigHandler.getfloat("mx_grad_nrm", "model_param")
    batch_size = ConfigHandler.getint("batch_size", "model_param")
    m = ConfigHandler.getint("layer_size", "model_param")
    num_samples = ConfigHandler.getint("num_samples", "model_param")
    use_lstm = ConfigHandler.get_boolean("use_lstm", "model_param")

    model = Seq2SeqModelTF(
        src_vocab_size=vocab_size_enc, tgt_vocab_size=vocab_size_dec, buckets=BUCKETS, m=m, num_layers=num_layers,
        mx_grad_nrm=mx_grad_nrm, batch_size=batch_size, lr=lr, model_name="seq2seq", save_dir="train_log",
        use_lstm=use_lstm, num_samples=num_samples, fwd_only=False
    )

    mode = ConfigHandler.get("train_mode", "model_param")

    if mode == "train":
        logging.info("mode: training")

        test_every = ConfigHandler.getint("test_every", "model_param")
        max_train_data_size = ConfigHandler.getint("max_train_data_size", "model_param")

        model.fit(
            enc_train, dec_train, enc_dev, dec_dev, max_train_data_size=max_train_data_size, test_every=test_every
        )
    elif mode == "test":
        logging.info("mode: testing")
        model.restore_latest_model()
        logging.info("beginning conversation, your turn first ({} to exit)".format(EXIT_PHRASE))

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            if sentence == EXIT_PHRASE:
                logging.info("ending conversation, have a good day!")
                break

            model.test(sentence)

            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


if __name__ == "__main__":
    main()
