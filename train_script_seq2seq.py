import sys
import logging

import data_util

from Seq2SeqModelTF import Seq2SeqModelTF

__author__ = "roopal_garg"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s : %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

BUCKETS = data_util.BUCKETS


def main():
    logging.info("preparing data")
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_util.prepare_datasets()

    logging.info("initializing the model")
    model = Seq2SeqModelTF(
        src_vocab_size=20000, tgt_vocab_size=20000, buckets=BUCKETS, m=256, num_layers=3, mx_grad_nrm=5.0, batch_size=64, lr=0.0001, model_name="seq2seq",
        save_dir="train_log", use_lstm=False, num_samples=512, fwd_only=False
    )

    "beginning training"
    model.fit(
        enc_train, dec_train, enc_dev, dec_dev, max_train_data_size=640, test_every=100
    )

if __name__ == "__main__":
    main()