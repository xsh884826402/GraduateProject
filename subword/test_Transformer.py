import os
import tensorflow as tf
from Transformer import Transformer, Config
from data_process import train_dev_split
from data_process import load_data




def main():

    config = Config()
    config.batch_size = 1024
    config.hidden_size = 64
    config.vocab_size = 26
    config.embed_size = 300
    config.max_epochs = 40
    config.label_kinds = 2
    config.if_train = True
    config.if_test = False
    config.is_biLSTM = True
    config.max_seqlen = 20

    config.original_file = '../data/input_word_label.txt'
    config.train_file = '../data/train.txt'
    config.dev_file = '../data/dev.txt'
    config.vocab_file = '../data/vocab.txt'
    config.model_path = 'models/Transformer_softmax/'

    config.split_ratio = 0.8

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
            (X_test, y_test) = load_data(config.dev_file)

            if len(X_test) < config.batch_size:
                for i in range(0, config.batch_size - len(X_test)):
                    X_test.append([0])
                    y_test.append([0])

            seq_len_test = list(map(lambda x: len(x), X_test))

            print('Target to special model to test')
            test_model = os.path.join(config.model_path, "models_epoch10")
            print('Start do predicting...')
            model.test(sess, test_model, X_test, y_test, seq_len_test,
                             config.vocab_file, config.model_path + 'result/')



    print('Success for preparing data')


if __name__ == '__main__':
    main()