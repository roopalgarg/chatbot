# Tensorflow Chatbot


The bot is modeled using Tensorflow's seq2seq module with attention. The architecture uses 3 layers which can be configured to be either LSTM or GRU based.

The input encoded sentences and the decoded output lengths are split into buckets of fixed lenghts as defined in the config file.

The output is trimmed if an EOS token is observed.

The vocab for the encoder and decoder is trimmed to 20000 words each.


The code is a re-wright from : https://github.com/llSourcell/tensorflow_chatbot  
