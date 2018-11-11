import time, datetime, os
import tensorflow as tf
from Dataset import Dataset
import numpy as np

""" 
Original taken from https://github.com/zhangluoyang/cnn-for-auto-event-extract
"""


class Model():
    def __init__(self,
                 sentence_length=30,
                 num_labels=10,
                 windows=3,
                 vocab_size=2048,
                 word_embedding_size=100,
                 pos_embedding_size=10,
                 filter_sizes=[3, 4, 5],
                 filter_num=200,
                 batch_size=10
                 ):
        """
        :param sentence_length
        :param num_labels
        :param windows
        :param vocab_size
        :param word_embedding_size
        :param pos_embedding_size
        :param filter_sizes
        :param filter_num
        """
        # [batch_size, sentence_length]
        input_x = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_x")
        self.input_x = input_x
        # [batch_size, num_labels]
        input_y = tf.placeholder(tf.float32, shape=[batch_size, num_labels], name="input_y")
        self.input_y = input_y
        # trigger distance vector
        # example: [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        input_t_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_t_pos")
        self.input_t_pos = input_t_pos
        # argument candidates distance vector
        # example: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        input_c_pos = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="input_c_pos")
        self.input_c_pos = input_c_pos
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob
        with tf.device('/cpu:0'), tf.name_scope("word_embedding_layer"):
            # [vocab_size, embedding_size]
            W_text = tf.Variable(tf.random_normal(shape=[vocab_size, word_embedding_size], mean=0.0, stddev=0.5), name="word_table")

            input_word_vec = tf.nn.embedding_lookup(W_text, input_x)
            input_t_pos_t = input_t_pos + (sentence_length - 1)
            Tri_pos = tf.Variable(
                tf.random_normal(shape=[2 * (sentence_length - 1) + 1, pos_embedding_size], mean=0.0, stddev=0.5),
                name="tri_pos_table")
            input_t_pos_vec = tf.nn.embedding_lookup(Tri_pos, input_t_pos_t)
            input_c_pos_c = input_c_pos + (sentence_length - 1)
            Can_pos = tf.Variable(
                tf.random_normal(shape=[2 * (sentence_length - 1) + 1, pos_embedding_size], mean=0.0, stddev=0.5),
                name="candidate_pos_table")
            input_c_pos_vec = tf.nn.embedding_lookup(Can_pos, input_c_pos_c)
            # print input_t_pos_vec
            # print input_c_pos_vec
            # The feature of the distance and the word features of the sentence constitute a collated feature as an input to the convolutional neural network.
            # [batch_size, sentence_length, word_embedding_size+2*pos_size]
            input_sentence_vec = tf.concat(2, [input_word_vec, input_t_pos_vec, input_c_pos_vec])
            # CNN supports 4d input, so increase the one-dimensional vector to indicate the number of input channels.
            input_sentence_vec_expanded = tf.expand_dims(input_sentence_vec, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/cpu:0'), tf.name_scope('conv-maxpool-%s' % filter_size):
                # The current word and context of the sentence feature considered here
                filter_shape = [filter_size, word_embedding_size + 2 * pos_embedding_size, 1, filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                # Convolution operation
                conv = tf.nn.conv2d(
                    input_sentence_vec_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maximize pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sentence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = filter_num * len(filter_sizes)
        # The number of all filters used (number of channels output)
        h_pool = tf.concat(3, pooled_outputs)
        # print h_pool
        # Expand to the next level classifier
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # print h_pool_flat
        lexical_vec = tf.reshape(input_word_vec, shape=(-1, sentence_length * word_embedding_size))
        # Combine lexical level features and sentence level features
        # [batch_size, num_filters_total] + [batch_size, sentence_length*word_embedding_size]
        all_input_features = tf.concat(1, [lexical_vec, h_pool_flat])
        # The overall classifier goes through a layer of dropout and then into softmax
        with tf.device('/cpu:0'), tf.name_scope('dropout'):
            all_features = tf.nn.dropout(all_input_features, dropout_keep_prob)
        # print all_features
        # Classifier
        with tf.device('/cpu:0'), tf.name_scope('softmax'):
            W = tf.Variable(tf.truncated_normal([num_filters_total + sentence_length * word_embedding_size, num_labels], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="b")
            scores = tf.nn.xw_plus_b(all_features, W, b, name="scores")
            predicts = tf.arg_max(scores, dimension=1, name="predicts")
            self.scores = scores
            self.predicts = predicts
        # print scores
        # print input_y
        # Cost function of the model
        with tf.device('/cpu:0'), tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
            loss = tf.reduce_mean(entropy)
            self.loss = loss
        # Accuracy is used for each training session
        with tf.device('/cpu:0'), tf.name_scope("accuracy"):
            correct = tf.equal(predicts, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
            self.accuracy = accuracy


batch_size = 30
max_sequence_length = 80
windows = 3  # The size of the selected context window
dataset = Dataset(batch_size=batch_size, max_sequence_length=max_sequence_length, windows=windows)

# parameters of the neural network model
sentence_length = max_sequence_length
num_labels = len(dataset.all_labels)
vocab_size = len(dataset.all_words)
word_embedding_size = 100
pos_embedding_size = 10
filter_sizes = [3, 4, 5]
filter_num = 100
lr = 1e-3
num_epochs = 20
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = Model(sentence_length=sentence_length,
                      num_labels=num_labels,
                      vocab_size=vocab_size,
                      word_embedding_size=word_embedding_size,
                      pos_embedding_size=pos_embedding_size,
                      filter_sizes=filter_sizes,
                      filter_num=filter_num,
                      batch_size=batch_size)

        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "model_01", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
        sess.run(tf.initialize_all_variables())


        def train_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                # model.input_t:input_t,
                # model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob,
            }
            _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


        def eval_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
            feed_dict = {
                model.input_x: input_x,
                model.input_y: input_y,
                # model.input_t:input_t,
                # model.input_c:input_c,
                model.input_t_pos: input_t_pos,
                model.input_c_pos: input_c_pos,
                model.dropout_keep_prob: dropout_keep_prob,
            }
            accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
            from sklearn.metrics import classification_report
            print("eval accuracy:{}".format(accuracy))
            print("input_y : ", [np.argmax(item) for item in input_y], ', predicts :', predicts)
            # print(classification_report(input_y, predicts, target_names=dataset.all_labels))
            print(classification_report([np.argmax(item) for item in input_y], predicts))
            return predicts


        for epoch in range(num_epochs):
            print('epoch: {}/{}'.format(epoch+1, num_epochs))
            for j in range(len(dataset.train_instances) // batch_size):
                x, t, c, y, pos_c, pos_t, _ = dataset.next_train_data()
                train_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t, dropout_keep_prob=0.8)

            if epoch % 3 == 0:
                x, t, c, y, pos_c, pos_t, _ = dataset.eval_data()
                eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t, dropout_keep_prob=1.0)

        print("----test results---------------------------------------------------------------------")
        x, t, c, y, pos_c, pos_t, _ = dataset.eval_data()
        predicts = eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c, input_t_pos=pos_t, dropout_keep_prob=1.0)

        for i in range(len(x)):
            print("Input data：{}".format(", ".join(map(lambda h: dataset.all_words[h], x[i]))))
            print("Trigger word：{}".format(", ".join(map(lambda h: dataset.all_words[h], t[i]))))
            print("Candidate：{}".format(", ".join(map(lambda h: dataset.all_words[h], c[i]))))
            print("Prediction:{}".format(predicts[i]))
            print("-------------------------------------------------------------------------")
