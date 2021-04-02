import os
import tensorflow as tf
from Transformer_softmax_sin_position import Transformer, Config
from data_process import train_dev_split
from data_process import load_data
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# class Config(object):
#     def __init__(self):
#         self.batch_size = 1024
#         self.hidden_size = 64
#         self.vocab_size = 26git
#         self.embed_size = 512
#         self.max_epochs = 40
#         self.label_kinds = 2
#         self.if_train = True
#         self.if_test = False
#         self.is_biLSTM = True
#         self.max_seqlen = 20
#
#         self.original_file = './data/input_word_label.txt'
#         self.train_file = './data/train.txt'
#         self.dev_file = './data/dev.txt'
#         self.vocab_file = 'data/vocab.txt'
#         self.model_path = 'models/bilstm/'
#
#         self.split_ratio = 0.8


def main():

    config = Config()
    config.batch_size = 1024
    config.hidden_size = 64
    config.vocab_size = 26
    config.embed_size = 320
    config.max_epochs = 300
    config.label_kinds = 2
    config.if_train = True
    config.if_test = False
    config.is_biLSTM = True
    config.max_seqlen = 20

    config.original_file = '../data/most_frequent_words_label.txt'
    config.train_file = '../data/most_frequent_words_label_train.txt'
    config.dev_file = '../data/most_frequent_words_label_dev'
    config.vocab_file = '../data/vocab.txt'
    config.model_path = 'models/Transformer_softmax_sin_position/'

    config.split_ratio = 0.8
    config.LSTM_dropout = 0.5
    print('Prepare data for train and dev ... ')
    train_dev_split(config.original_file, config.train_file, config.dev_file,
                    config.vocab_file, config.split_ratio)
    print('Prepare data sucessfully!')

    model = Transformer(config)

    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True) ##每个gpu占用0.8                                                                              的显存
    tf_config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    with tf.Session(config=tf_config) as sess:
        if config.if_train:
            init=tf.global_variables_initializer()
            sess.run(init)
            (X_train, y_train) = load_data(config.train_file)

            if len(X_train) < config.batch_size:
                for i in range(0, config.batch_size - len(X_train)):
                    X_train.append([0])
                    y_train.append([0])

            seq_len_train = list(map(lambda x: len(x), X_train))
            model.train_epoch(sess, config.train_file, X_train,
                                  y_train, seq_len_train,
                                  config.model_path)

    print('Success for preparing data')


if __name__ == '__main__':
    main()