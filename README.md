# Tensorflow Chatbot


The bot is modeled using Tensorflow's seq2seq module with attention. The architecture uses 3 layers which can be configured to be either LSTM or GRU based.
The input encoded sentence and the decoded output lenghts are split into buckets as defined in the config file.


The code is a re-wright from : https://github.com/llSourcell/tensorflow_chatbot  
