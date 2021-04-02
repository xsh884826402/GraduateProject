import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from keras_preprocessing import sequence
from data_process import write_preds, eval_metrics

warnings.filterwarnings("ignore")


# %%

# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):

    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0


class Config(object):


    split_ratio = 0.8
    sequenceLength = 20  # 取了所有序列长度的均值
    batchSize = 128


    numClasses = 2  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()


# %%

# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# %%

# 生成位置嵌入
def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)

    return np.array(embeddedPosition, dtype="float32")




# 模型构建


class Transformer(object):
    """
    Transformer Encoder 用于文本分类
    """

    def __init__(self, config, ):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.max_epochs = config.max_epochs
        self.label_kinds = config.label_kinds
        self.if_train = config.if_train
        self.dev_file = config.dev_file
        self.is_biLSTM = config.is_biLSTM
        self.max_seqlen = config.max_seqlen

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.max_seqlen], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None, config.max_seqlen], name="inputY")
        self.sequence_length_placeholder = tf.placeholder(tf.int32)

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.embeddedPosition = tf.placeholder(tf.float32, [None, config.sequenceLength, config.sequenceLength],
                                               name="embeddedPosition")

        self.config = config

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，在当前的数据集上表现效果更好。另一种
        # 就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。

        # with tf.name_scope("embedding"):
        #
        #     # 利用预训练的词向量初始化词嵌入矩阵
        #     self.W = tf.get_variable(name="Embedding",
        #                              shape=[self.vocab_size, self.embed_size])
        #     # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        #     self.embedded = tf.nn.embedding_lookup(self.W, self.inputX)
        #     self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.get_variable(name="Embedding",
                                     shape=[self.vocab_size, self.embed_size])
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded = tf.nn.embedding_lookup(self.W, self.inputX)

            ##initial state
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.embed_size//2,
                                                   forget_bias=0.0)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.embed_size//2,
                                                   forget_bias=0.0)
            if self.if_train:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                    cell_fw, output_keep_prob=(1 - self.dropoutKeepProb))
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                    cell_bw, output_keep_prob=(1 - self.dropoutKeepProb))

            self.initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            self.initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                self.embedded,
                initial_state_fw=self.initial_state_fw,
                initial_state_bw=self.initial_state_bw,
                sequence_length=self.sequence_length_placeholder)
            print('bilstm shape ',np.shape(outputs), np.shape(state))
            outputs = tf.concat(outputs, 2)
            self.lstm_outputs = tf.reshape(outputs, [self.batch_size, self.max_seqlen, self.embed_size])

        with tf.name_scope("transformer"):
            for i in range(config.model.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    # 维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX, queries=self.lstm_outputs,
                                                            keys=self.lstm_outputs)
                    # 维度[batch_size, sequence_length, embedding_size]
                    self.embeddedWords = self._feedForward(multiHeadAtt,
                                                           [config.model.filters,
                                                            config.embed_size])
                    print('in Transformer embeddword \n\n', np.shape(self.embeddedWords))
            outputs = tf.reshape(self.embeddedWords,
                                 [-1, config.max_seqlen, config.embed_size])

        outputSize = outputs.get_shape()[-1].value
        print("shape", np.shape(outputs),)

        #         with tf.name_scope("wordEmbedding"):
        #             self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
        #             self.wordEmbedded = tf.nn.embedding_lookup(self.W, self.inputX)

        #         with tf.name_scope("positionEmbedding"):
        #             print(self.wordEmbedded)
        #             self.positionEmbedded = self._positionEmbedding()

        #         self.embeddedWords = self.wordEmbedded + self.positionEmbedded

        #         with tf.name_scope("transformer"):
        #             for i in range(config.model.numBlocks):
        #                 with tf.name_scope("transformer-{}".format(i + 1)):

        #                     # 维度[batch_size, sequence_length, embedding_size]
        #                     multiHeadAtt = self._multiheadAttention(rawKeys=self.wordEmbedded, queries=self.embeddedWords,
        #                                                             keys=self.embeddedWords)
        #                     # 维度[batch_size, sequence_length, embedding_size]
        #                     self.embeddedWords = self._feedForward(multiHeadAtt, [config.model.filters, config.model.embeddingSize])

        #             outputs = tf.reshape(self.embeddedWords, [-1, config.sequenceLength * (config.model.embeddingSize)])

        #         outputSize = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            print('logit', np.shape(self.logits))


            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1,config.max_seqlen, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.inputY,
                                                                                                              [config.batch_size, config.max_seqlen]))

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

        self.globalStep = tf.Variable(0, name="globalStep", trainable=False)

        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        self.gradsAndVars = optimizer.compute_gradients(self.loss)
        # 将梯度应用到变量下，生成训练器
        self.trainOp = optimizer.apply_gradients(self.gradsAndVars, global_step=self.globalStep)

        self.summaryOp = tf.summary.merge_all()

    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = self.config.model.epsilon

        inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值

        numHeads = self.config.model.numHeads
        keepProp = self.config.model.keepProp

        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
                                  scaledSimilary)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
                                      maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络
        print('feed forwardinputs', np.shape(inputs), filters)
        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)
        print('shape', 'input shape', np.shape(inputs), 'output shape', np.shape(outputs))


        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded

    def padding_data(self, x, y, maxlen):
        x = np.array(x)
        x = sequence.pad_sequences(x,
                                   maxlen=maxlen,
                                   padding='post',
                                   truncating='post')

        y = sequence.pad_sequences(y,
                                   maxlen = maxlen,
                                   padding='post',
                                   truncating='post')
        return [x, y]


    def do_evaluation(self, sess, X, y, X_len, write_file):
        (X, y) = self.padding_data(X, y, self.max_seqlen)
        num_steps = len(X) // self.batch_size
        # init_state = sess.run([self.initial_state])
        x_test = []
        y_test = []
        y_refs = []
        mean_loss = []
        for step in list(range(num_steps)):
            input_batch = X[step * self.batch_size:(step + 1) *
                            self.batch_size]
            label_batch = y[step * self.batch_size:(step + 1) *
                            self.batch_size]
            seq_len = X_len[step * self.batch_size:(step + 1) *
                            self.batch_size]
            feed = {
                self.inputX: input_batch,
                self.inputY: label_batch,
                self.dropoutKeepProb: self.config.model.dropoutKeepProb,
                self.sequence_length_placeholder: seq_len,
                self.embeddedPosition: fixedPositionEmbedding(self.config.batch_size, self.config.max_seqlen)
            }

            pred, loss_step = sess.run(
                [self.predictions, self.loss], feed)
            # print('pred shape: ', pred.shape)

            y_test = y_test + pred.tolist()
            x_test = x_test + input_batch.tolist()
            y_refs = y_refs + label_batch.tolist()
            loss_step = np.mean(loss_step)
            mean_loss.append(loss_step)

        left = len(X) % self.batch_size
        input_batch = X[-self.batch_size:]
        label_batch = y[-self.batch_size:]
        seq_len = X_len[-self.batch_size:]
        feed = {
            self.inputX: input_batch,
            self.inputY: label_batch,
            self.dropoutKeepProb: self.config.model.dropoutKeepProb,
            self.sequence_length_placeholder: seq_len,
            self.embeddedPosition: fixedPositionEmbedding(self.config.batch_size, self.config.max_seqlen)
        }
        pred = sess.run(self.predictions, feed)
        y_test = y_test + pred.tolist()[-left:]
        x_test = x_test + input_batch.tolist()[-left:]
        y_refs = y_refs + label_batch.tolist()[-left:]
        print('in do evaluation',type(y_test),y_test[1])
        write_preds(y_test, write_file)

        return x_test, y_test

    def test(self,
             sess,
             model,
             X_test,
             y_test,
             X_len_test,
             dict_file="./data/char.dict",
             resultFile="result.txt"):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, model)  #load model
        write_file = resultFile+'pred_' + os.path.basename(model)
        print('Start do eval...')
        if not os.path.exists(write_file):
            xtest, ytest = self.do_evaluation(sess, X_test, y_test,
                                              list(X_len_test), write_file)
        eval_metrics(write_file, modelname=model, refs_path=self.config.dev_file)


    def train_epoch(self,
                    sess,
                    train_file,
                    X_train,
                    y_train,
                    X_len,
                    model_path,
                    restore_model=False):


        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=self.config.max_epochs)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        ckpt = tf.train.get_checkpoint_state(model_path)


        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        summary_writer = tf.summary.FileWriter("log/", sess.graph)

        for epoch in range(self.max_epochs):
            print('%d Epoch starts, Training....' % (epoch))
            start_time = time.time()
            mean_loss = []
            # state = sess.run([self.initial_state])

            shuf_step = list(range(len(X_train) // self.batch_size))
            np.random.shuffle(shuf_step)
            batch_count = 0
            for step in shuf_step:
                # generate the data feed dict
                input_batch = X_train[step * self.batch_size:(step + 1) *
                                      self.batch_size]
                label_batch = y_train[step * self.batch_size:(step + 1) *
                                      self.batch_size]

                seq_len = X_len[step * self.batch_size:(step + 1) *
                                self.batch_size]

                (input_batch,
                 label_batch) = self.padding_data(input_batch,
                                               label_batch,
                                               self.max_seqlen
                                               )
                seq_len = [min(x, self.max_seqlen) for x in seq_len]

                feed = {
                    self.inputX: input_batch,
                    self.inputY: label_batch,
                    self.dropoutKeepProb: self.config.model.dropoutKeepProb,
                    self.sequence_length_placeholder: seq_len,
                    self.embeddedPosition: fixedPositionEmbedding(self.config.batch_size, self.config.max_seqlen)
                }
                _, loss, predictions = sess.run(
                    [self.trainOp, self.loss, self.predictions],
                    feed)

                loss = np.sum(loss)
                mean_loss.append(loss)
                batch_count += 1
                if batch_count % 1000 == 0:
                    print('step %d / %d : time: %ds, loss : %f' %
                          (batch_count, len(X_train) // self.batch_size,
                           time.time() - start_time, np.mean(loss)))
                    mean_loss = []

            train_loss = np.mean(mean_loss)
            mean_loss = []
            print('epoch: {} loss: {}'.format(epoch, train_loss))
            tf.summary.scalar('loss', train_loss)
            tf.summary.merge_all()
            print('epoch_time: %ds' % (time.time() - start_time))
            save_path = saver.save(
                sess, os.path.join(model_path, "models_epoch" + str(epoch)))




# %%

# 训练模型

# 生成训练集和验证集


# embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength)

# 定义计算图
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#     session_conf.gpu_options.allow_growth = True
#     session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
#
#     sess = tf.Session(config=session_conf)
#
#     # 定义会话
#     with sess.as_default():
#         transformer = Transformer(config, wordEmbedding)
#
#         globalStep = tf.Variable(0, name="globalStep", trainable=False)
#         # 定义优化函数，传入学习速率参数
#         optimizer = tf.train.AdamOptimizer(config.training.learningRate)
#         # 计算梯度,得到梯度和变量
#         gradsAndVars = optimizer.compute_gradients(transformer.loss)
#         # 将梯度应用到变量下，生成训练器
#         trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
#
#         # 用summary绘制tensorBoard
#         gradSummaries = []
#         for g, v in gradsAndVars:
#             if g is not None:
#                 tf.summary.histogram("{}/grad/hist".format(v.name), g)
#                 tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#
#         outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
#         print("Writing to {}\n".format(outDir))
#
#         lossSummary = tf.summary.scalar("loss", transformer.loss)
#         summaryOp = tf.summary.merge_all()
#
#         trainSummaryDir = os.path.join(outDir, "train")
#         trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
#
#         evalSummaryDir = os.path.join(outDir, "eval")
#         evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
#
#         # 初始化所有变量
#         saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
#
#         # 保存模型的一种方式，保存为pb文件
#         savedModelPath = "../model/transformer/savedModel"
#         if os.path.exists(savedModelPath):
#             os.rmdir(savedModelPath)
#         builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
#
#         sess.run(tf.global_variables_initializer())
#
#
#         def trainStep(batchX, batchY):
#             """
#             训练函数
#             """
#             feed_dict = {
#                 transformer.inputX: batchX,
#                 transformer.inputY: batchY,
#                 transformer.dropoutKeepProb: config.model.dropoutKeepProb,
#                 transformer.embeddedPosition: embeddedPosition
#             }
#             _, summary, step, loss, predictions = sess.run(
#                 [trainOp, summaryOp, globalStep, transformer.loss, transformer.predictions],
#                 feed_dict)
#
#             if config.numClasses == 1:
#                 acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
#
#
#             elif config.numClasses > 1:
#                 acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
#                                                               labels=labelList)
#
#             trainSummaryWriter.add_summary(summary, step)
#
#             return loss, acc, prec, recall, f_beta
#
#
#         def checkStateStep(batchX,batchY):
#             feed_dict = {
#                 transformer.inputX: batchX,
#                 transformer.inputY: batchY,
#                 transformer.dropoutKeepProb: config.model.dropoutKeepProb,
#                 transformer.embeddedPosition: embeddedPosition
#             }
#             X = sess.run(
#                 [transformer.inputX],feed_dict=feed_dict)
#             return X
#
#
#         def devStep(batchX, batchY):
#             """
#             验证函数
#             """
#             feed_dict = {
#                 transformer.inputX: batchX,
#                 transformer.inputY: batchY,
#                 transformer.dropoutKeepProb: 1.0,
#                 transformer.embeddedPosition: embeddedPosition
#             }
#             summary, step, loss, predictions = sess.run(
#                 [summaryOp, globalStep, transformer.loss, transformer.predictions],
#                 feed_dict)
#
#             if config.numClasses == 1:
#                 acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
#
#
#             elif config.numClasses > 1:
#                 acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
#                                                               labels=labelList)
#
#             trainSummaryWriter.add_summary(summary, step)
#
#             return loss, acc, prec, recall, f_beta
#
#
#         for i in range(config.training.epoches):
#             # 训练模型
#             print("start training model")
#             for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
#                 loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])
#                 print('CheckState', checkStateStep(batchTrain[0], batchTrain[1]), np.shape(checkStateStep(batchTrain[0], batchTrain[1])))
#
#                 currentStep = tf.train.global_step(sess, globalStep)
#                 print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
#                     currentStep, loss, acc, recall, prec, f_beta))
#                 if currentStep % config.training.evaluateEvery == 0:
#                     print("\nEvaluation:")
#
#                     losses = []
#                     accs = []
#                     f_betas = []
#                     precisions = []
#                     recalls = []
#
#                     for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
#                         loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
#                         losses.append(loss)
#                         accs.append(acc)
#                         f_betas.append(f_beta)
#                         precisions.append(precision)
#                         recalls.append(recall)
#
#                     time_str = datetime.datetime.now().isoformat()
#                     print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
#                                                                                                          currentStep,
#                                                                                                          mean(losses),
#                                                                                                          mean(accs),
#                                                                                                          mean(
#                                                                                                              precisions),
#                                                                                                          mean(recalls),
#                                                                                                          mean(f_betas)))
#
#                 if currentStep % config.training.checkpointEvery == 0:
#                     # 保存模型的另一种方法，保存checkpoint文件
#                     path = saver.save(sess, "../model/Transformer/model/my-model", global_step=currentStep)
#                     print("Saved model checkpoint to {}\n".format(path))
#
#         inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
#                   "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}
#
#         outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}
#
#         prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
#                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#         legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
#         builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
#                                              signature_def_map={"predict": prediction_signature},
#                                              legacy_init_op=legacy_init_op)
#
#         builder.save()
