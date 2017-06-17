# -*- coding: utf-8 -*-
import logging

from Tokens import Tokens
from config.ConfigHandler import ConfigHandler
from nltk.tokenize import word_tokenize

from tensorflow.python.platform import gfile


def create_vocab(vocab_path, data_path, max_vocab_size):
    if not gfile.Exists(vocab_path):
        logging.info("generating vocab at {} from {}".format(vocab_path, data_path))

        dict_vocab = dict()
        with gfile.GFile(data_path, mode="rb") as fp:
            line_counter = 0
            for line in fp:
                line_counter += 1
                if line_counter % 100000 == 0:
                    logging.info("\t\tprocessing line {}".format(line_counter))

                tokens = [word.rstrip('-') for word in word_tokenize(line.decode('utf-8'))]
                for word in tokens:
                    if word in dict_vocab:
                        dict_vocab[word] += 1
                    else:
                        dict_vocab[word] = 1

        vocab_list = Tokens.list_tokens + sorted(dict_vocab, key=dict_vocab.get, reverse=True)
        logging.info("size of vocab: {}".format(len(vocab_list)))

        if len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]

        with gfile.GFile(vocab_path, mode="wb") as vocab_fp:
            for word in vocab_list:
                vocab_fp.write(word + "\n")
    else:
        logging.info("skipping create vocab: {}".format(vocab_path))


def get_wrd2idx(vocab_path):
    if gfile.Exists(vocab_path):
        list_vocab = list()

        with gfile.GFile(vocab_path, mode="rb") as f:
            list_vocab.extend(f.readlines())

        list_vocab = [line.strip() for line in list_vocab]
        word2idx = dict([(x, y) for (y, x) in enumerate(list_vocab)])

        return word2idx

    else:
        raise ValueError("vocab file not found: {}".format(vocab_path))


def sentence_to_token_ids(sentence, wrd2idx):
    words = word_tokenize(sentence)
    return [wrd2idx.get(w, Tokens.UNK.idx) for w in words]


def data_to_token_ids(data_path, target_path, vocab_path):
    if not gfile.Exists(target_path):

        logging.info("tokenizing data in {}".format(data_path))

        word2idx, _ = get_wrd2idx(vocab_path)

        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:

                counter = 0

                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        logging.info("\t\ttokenizing line: {}".format(counter))

                    token_ids = sentence_to_token_ids(line, word2idx)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
    else:
        logging.info("skipping data to token ids: {}".format(target_path))


def prepare_datasets():
    """

    :return:
    """

    """create the vocabs for enc and dec"""
    enc_vocab_path = ConfigHandler.get("vocab_enc", "data_path")
    train_enc = ConfigHandler.get("train_enc", "data_path")
    enc_vocab_size = ConfigHandler.getint("vocab_size_enc", "model_param")

    dec_vocab_path = ConfigHandler.get("vocab_dec", "data_path")
    train_dec = ConfigHandler.get("train_dec", "data_path")
    dec_vocab_size = ConfigHandler.getint("vocab_size_dec", "model_param")

    create_vocab(enc_vocab_path, train_enc, enc_vocab_size)
    create_vocab(dec_vocab_path, train_dec, dec_vocab_size)

    """create the tokens ids for training data"""
    enc_train_idx_path = "{}_{}.idx".format(train_enc, enc_vocab_size)
    data_to_token_ids(train_enc, enc_train_idx_path, enc_vocab_path)

    dec_train_idx_path = "{}_{}.idx".format(train_dec, dec_vocab_size)
    data_to_token_ids(train_dec, dec_train_idx_path, dec_vocab_path)

    """create the tokens ids for test/dev data"""
    test_enc = ConfigHandler.get("test_enc", "data_path")
    test_dec = ConfigHandler.get("test_dec", "data_path")

    enc_test_idx_path = "{}_{}.idx".format(test_enc, enc_vocab_size)
    data_to_token_ids(test_enc, enc_test_idx_path, enc_vocab_path)

    dec_test_idx_path = "{}_{}.idx".format(test_dec, dec_vocab_size)
    data_to_token_ids(test_dec, dec_test_idx_path, dec_vocab_path)

    return {
        "train": {
            "enc_path": enc_train_idx_path, "dec_path": dec_train_idx_path, "enc_vocab": enc_vocab_path,
            "dec_vocab": dec_vocab_path
        },
        "test": {
            "enc_path": enc_test_idx_path, "dec_path": dec_test_idx_path
        }
    }

prepare_datasets()