from collections import namedtuple

__author__ = "roopal_garg"


class Tokens:
    _TOKEN = namedtuple("TOKEN", 'value idx')
    PAD = _TOKEN(value="<PAD>", idx=0)
    GO = _TOKEN(value="<GO>", idx=1)
    EOS = _TOKEN(value="<EOS>", idx=2)
    UNK = _TOKEN(value="<UNK>", idx=3)

    list_tokens = [PAD.value, GO.value, EOS.value, UNK.value]
